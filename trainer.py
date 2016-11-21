import vgg
import tensorflow as tf
from scipy.misc import imread, imsave, imresize
import numpy as np
import logging
import argparse
from sklearn.decomposition import PCA
import os

class TensorPCA(PCA):
    def transform_tensor(self, X):
        ''' X : [n_samples, channels]
        mean_ : [channels]
        componenets_ : [n_components, channels]
        '''
        if self.mean_ is not None:
            X = X - np.expand_dims(self.mean_, axis=0)
        X_transformed = tf.matmul(X, self.components_.astype(np.float32), transpose_b=True)
        if self.whiten:
            # X_transformed /= np.sqrt(self.explained_variance_)
            raise NotImplementedError
        return X_transformed

class StyleTransform:

    content_layer = 'relu4_2'
    style_layers = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
    kwargs_list = ('loss_weight', 'pca')
    pca = False
    loss_weight = {
        'content' : 1,
        'style' : 50,
        'regular' : 5e-2
    }

    def __init__(self, content_image, style_image, network_path = "imagenet-vgg-verydeep-19.mat", **kwargs):

        for k in kwargs:
            if k in self.kwargs_list:
                setattr(self, k, kwargs[k])

        with tf.Graph().as_default():
            image = tf.placeholder(tf.float32, shape=(1, None, None, 3))
            net, self.mean_pixel = vgg.net(network_path, image)

            self.style_feat_pca = dict()
            if self.pca:
                with tf.Session() as sess:
                    style_feats = sess.run([net[layer] for layer in self.style_layers], feed_dict={image : self.preprocess(style_image)})
                for layer, feats in zip(self.style_layers, style_feats):
                    # [1, H, W, C]
                    print(feats.shape)
                    channels = feats.shape[3]
                    model = TensorPCA(n_components=int(channels ** (5.0 / 6)))
                    model.fit(np.reshape(feats, [-1, channels]))
                    self.style_feat_pca[layer] = model

            style_grams = [ self.opr_gram(net[layer], layer) for layer in self.style_layers]

            with tf.Session() as sess:
                content_feat = sess.run(net[self.content_layer], feed_dict={image : self.preprocess(content_image)})
                style_feats = sess.run(style_grams, feed_dict={image : self.preprocess(style_image)})

            # print(content_feat.shape)
            # for style_feat in style_feats:
            #     print(style_feat.shape)

        with tf.Graph().as_default():
            # image : shape (1, H, W, C)
            image = tf.Variable(self.preprocess(content_image), dtype=tf.float32)
            self.output = image
            net, self.mean_pixel = vgg.net(network_path, image)

            self.subloss = dict()

            generate_content_feat = net[self.content_layer]
            self.subloss['content'] = self.opr_l2_loss(generate_content_feat - content_feat) / tf.cast(tf.shape(content_feat)[3], tf.float32)

            generate_style_feats = [ self.opr_gram(net[layer], layer) for layer in self.style_layers]
            feature_map_sizes = [ self.opr_map_size(net[layer]) for layer in self.style_layers]
            self.subloss['style'] = sum( map(lambda x, y, s: self.opr_l2_loss(x - y, normed=True) * s, generate_style_feats, style_feats, feature_map_sizes) )

            self.subloss['regular'] = self.opr_l2_loss(image[:, 1 :] - image[:, :-1]) + self.opr_l2_loss(image[:, :, 1:] - image[:, :, :-1])

            self.loss = sum([self.subloss[key] * self.loss_weight[key] for key in self.subloss])

            self.learning_rate = tf.placeholder(tf.float32)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.995)
            self.train_op = optimizer.minimize(self.loss)

            self.sess = tf.Session()
            self.sess.run(tf.initialize_all_variables())

        logging.info('Build up trainer, loss_weight={}'.format(self.loss_weight))


    def preprocess(self, image):
        return np.expand_dims(vgg.preprocess(image, self.mean_pixel), axis=0)

    def unprocess(self, image):
        return vgg.unprocess(image, self.mean_pixel)[0]

    def opr_gram(self, vin, layer):
        vin = tf.reshape(vin, [-1, tf.shape(vin)[3] ] )
        if layer in self.style_feat_pca:
            vin = self.style_feat_pca[layer].transform_tensor(vin)
            vout = tf.matmul(vin, vin, transpose_a=True) / tf.cast(tf.shape(vin)[0], tf.float32)
            vout = vout - tf.diag(tf.diag_part(vout))
        else:
            vout = tf.matmul(vin, vin, transpose_a=True) / tf.cast(tf.shape(vin)[0], tf.float32)
        return vout

    def opr_map_size(self, vin):
        ''' vin : [1, H, W, C]
        '''
        shape = tf.shape(vin)
        return tf.cast(shape[1] * shape[2], tf.float32)

    def opr_l2_loss(self, vin, normed=False):
        reduce_opr = tf.reduce_mean if normed else tf.reduce_sum
        return reduce_opr(vin ** 2)

    def train(self, num_iterations, learning_rate = 1, logging_iterations = 10, dump_image = None):
        logging.info('Start training with learing_rate = {}'.format(learning_rate))

        for i in range(num_iterations):
            self.sess.run(self.train_op, feed_dict={self.learning_rate : learning_rate})
            if (i + 1) % logging_iterations == 0:
                subloss = self.sess.run(self.subloss)
                subloss['loss'] = self.sess.run(self.loss)

                subloss_str = ', '.join(['%s = %.4e' % (k, subloss[k]) for k in ['content', 'style', 'regular']])

                logging.debug('Iteration %d / %d: %s' % (i + 1, num_iterations, subloss_str) )
                if dump_image is not None:
                    self.export(os.path.join(dump_image, 'result-%d.jpg') % (i + 1))

    def export(self, filename):
        image = self.sess.run(self.output)
        image = self.unprocess(image)
        image = np.clip(image, 0, 255).astype(np.uint8)
        imsave(filename, image)

def reshape(image, short_edge = 500):
    height, width = image.shape[:2]
    fraction = float(short_edge) / min(height, width)
    return imresize(image, fraction, interp='cubic')

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s]%(levelname)s %(message)s',
        level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str, default="examples/1-content.jpg")
    parser.add_argument('--style', type=str, default="examples/1-style.jpg")
    parser.add_argument('--output', type=str, default='result.jpg')
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--reshape', action='store_true')
    parser.add_argument('--dump', type=str, default=None)
    parser.add_argument('--pca', action='store_true')

    args = parser.parse_args()

    content_path = args.content
    content_image = imread(content_path).astype(np.float32)

    style_path = args.style
    style_image = imread(style_path).astype(np.float32)

    if args.reshape:
        content_image = reshape(content_image)
        style_image = reshape(style_image)

    if args.dump is not None:
        if not os.path.exists(args.dump):
            os.mkdir(args.dump)

    model = StyleTransform(content_image, style_image, pca=args.pca)
    model.train(args.iterations, learning_rate = 2, logging_iterations = 50, dump_image=args.dump)
    model.export(args.output)
