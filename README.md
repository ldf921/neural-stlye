# neural-style

This project follows the [implmentation](https://github.com/anishathalye/neural-style) of neural style by Anish Athalye. I reuse their pretrain VGG model, and code *vgg.py* for building VGG-19 model. The main file *trainer.py* is implemented purely by myself, with reference to their code *stylize.py* and *neural_style.py*.

## Running

`python3 trainer.py --content <content file> --styles <style file> --output <output file>`

## Requirements

* python3
* [TensorFlow](https://www.tensorflow.org/versions/master/get_started/os_setup.html#download-and-setup)
* [NumPy](https://github.com/numpy/numpy/blob/master/INSTALL.rst.txt)
* [SciPy](https://github.com/scipy/scipy/blob/master/INSTALL.rst.txt)
* [Pillow](http://pillow.readthedocs.io/en/3.3.x/installation.html#installation)
* [Pre-trained VGG network][net] (MD5 `8ee3263992981a1d26e73b3ca028a123`), put it in the top level of this repository

## License

Copyright (c) 2015-2016 Dingfeng, Liu. Released under GPLv3. See
[LICENSE.txt][license] for details.
