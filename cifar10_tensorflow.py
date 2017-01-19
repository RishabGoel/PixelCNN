from keras.datasets import cifar10
import numpy
from generic_utils import *
from gated_pixel_cnn import GatedPixelCNN
import tensorflow as tf
Q_LEVELS = 256

(X_train_r, _), (X_test_r, _) = cifar10.load_data()
X_train_r = X_train_r[:25]
X_test_r = X_test_r[:100]
print X_train_r.shape
X_train_r = upscale_images(downscale_images(X_train_r, 256), Q_LEVELS) 
X_test_r = upscale_images(downscale_images(X_test_r, 256), Q_LEVELS)

X_train = downscale_images(X_train_r, Q_LEVELS - 1)
X_test = downscale_images(X_test_r, Q_LEVELS - 1)
X = tf.placeholder(tf.float32, [25, 32, 32, 3])
X_shape = tf.placeholder(tf.float32, [4])
a = GatedPixelCNN(X, X_train.shape)
b = a.get_output()
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	print sess.run(b, feed_dict = {X : X_train}).shape
	print sess.run(b, feed_dict = {X : X_train})