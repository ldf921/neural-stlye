import vgg
import tensorflow as tf
from scipy.misc import imread, imsave, imresize
import numpy as np
import logging
import argparse

class StyleTransform:

    content_layer = 'relu4_2'
    style_layers = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
    kwargs_list = ('loss_weight')
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
            style_grams = [ self.opr_gram(net[layer]) for layer in self.style_layers]

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

            generate_style_feats = [ self.opr_gram(net[layer]) for layer in self.style_layers]
            self.subloss['style'] = sum( map(lambda x, y: self.opr_l2_loss(x - y, normed=True), generate_style_feats, style_feats) )

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

    def opr_gram(self, vin):
        vin = tf.reshape(vin, [-1, tf.shape(vin)[3] ] )
        vout = tf.matmul(vin, vin, transpose_a=True)
        return vout

    def opr_l2_loss(self, vin, normed=False):
        reduce_opr = tf.reduce_mean if normed else tf.reduce_sum
        return reduce_opr(vin ** 2)

    def train(self, num_iterations, learning_rate = 1, logging_iterations = 10):
        logging.info('Start training with learing_rate = {}'.format(learning_rate))

        for i in range(num_iterations):
            self.sess.run(self.train_op, feed_dict={self.learning_rate : learning_rate})
            if (i + 1) % logging_iterations == 0:
                subloss = self.sess.run(self.subloss)
                subloss['loss'] = self.sess.run(self.loss)

                subloss_str = ', '.join(['%s = %.4e' % (k, subloss[k]) for k in ['content', 'style', 'regular']])

                logging.debug('Iteration %d / %d: %s' % (i + 1, num_iterations, subloss_str) )
                # self.export('img/result-%d.jpg' % (i + 1))

    def export(self, filename):
        image = self.sess.run(self.output)
        image = self.unprocess(image)
        image = np.clip(image, 0, 255).astype(np.uint8)
        imsave(filename, image)

def reshape(image, short_edge = 600):
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

    args = parser.parse_args()

    content_path = args.content
    content_image = imread(content_path).astype(np.float32)

    style_path = args.style
    style_image = imread(style_path).astype(np.float32)

    if args.reshape:
        content_image = reshape(content_image)
        style_image = reshape(style_image)

    model = StyleTransform(content_image, style_image)
    model.train(args.iterations, learning_rate = 2, logging_iterations = 50)
    model.export(args.output)
