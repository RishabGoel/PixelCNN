"""
This trains a generative PixelCNN model on mnist.

"""

import os
import sys
import time
import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

parser = argparse.ArgumentParser()

# Data IO
parser.add_argument('-i', '--data_dir', type=str, default='/tmp/pixelcnn/data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='/tmp/pixelcnn/save', help='Location for parameter checkpoints and samples')
parser.add_argument('-t', '--save_interval', type=int, default=20, help='Every how many epochs to write checkpoint/samples?')

# model
parser.add_argument('-a', '--activation', type = str, default = "relu", help = "Activation to be used for layers")
parser.add_argument('-l', '--layers', type = int, default = 12, help = "Number of layers in the network")
parser.add_argument('-fs', '--features', type = int, default = 32, help = "Number of features")
parser.add_argument('-ql', '--q_levels', type = int, default = None, help = "Q_levels")
# Optimization
parser.add_argument('-f', '--lr', type=float, default=0.001, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995, help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=200, help='Batch size during training per GPU')
parser.add_argument('-p', '--dropout_p', type=float, default=0.5, help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-x', '--max_epochs', type=int, default=5000, help='How many epochs to run in total?')

# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')

# print parser.parse_args()

class main(object):
	"""This class creates the PixelCNN model and generates the images"""

	def __init__(self):
		
		"""This methods reads the data"""
		data_object = load_digits()
		self.input_images = self.__reshape_data(data_object.data)[:1600]

		self.targets = data_object.target[:1600]
		self.images = data_object.images
		self.init = tf.global_variables_initializer()
		self.__saver = tf.train.Saver()

	def visualise_images(self): 
		"""This function helps to visualise the images"""
		# plt.gray()
		plt.matshow(self.images[0])
		plt.show()

	def __get_next_batch(self, input_images, input_images_r, batch_no):
		return (input_images[self.__batch_size*(batch_no): self.__batch_size*(batch_no+1)],
				input_images_r[self.__batch_size*(batch_no): self.__batch_size*(batch_no+1)])

	def __reshape_data(self, data):
		in_shape = data.shape
		return np.reshape(data, (in_shape[0], 8, 8, -1))

	
	def train(self, parser):
		"""
		This function creates the computation graph and trains a PixelCNN model
		"""
		self.out_dir = parser.save_dir
		input_shape = self.input_images.shape
		num_batches = input_shape[0] // parser.batch_size
		exists = False

		if os.path.isfile(save_dir+"model.ckpt") == False:
			init = tf.global_variables_initializer()
			X = tf.placeholder(tf.float32, [None, 8, 8, 1])
			X_r = tf.placeholder(tf.float32, [None, 8, 8, 1])
			# print "X", X.get_shape()
			pixelcnn = PixelCNN(X, [200, 8, 8, 1], parser.activation, parser.features,
												parser.q_levels, [7,3], parser.layers)
			pixelcnn_output = pixelcnn.get_output()
			p_o_s = pixelcnn_output.get_shape().as_list()
			pixelcnn_output = tf.reshape(pixelcnn_output, [p_o_s[0], p_o_s[1], p_o_s[2]])
			print pixelcnn_output.get_shape()
			output_prob = self.softmax(pixelcnn_output)
			
			print output_prob.get_shape(), X_r.get_shape(),"shapes"
			cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_prob, tf.reshape(X_r,[-1, 1])))
			optimizer = tf.train.AdamOptimizer(parser.lr).minimize(cost)
		else:
			exists = True
		
		with tf.Session() as sess:
			if exists == False:
				print "Model file not found. Initializing variables for a new model"
				sess.run(self.init)
			else:
				print "Model loading from file: ",parser.save_dir+"model.ckpt"
				self.__saver.restore()
			for i in range(parser.max_epochs):
				batch_no = 0
				while (batch_no < num_batches):
					x_train, x_train_r = self.__get_next_batch(self.input_images, self.input_images, batch_no)

					feed_dict = {
									X : x_train,
									X_r : x_train_r,
									input_dims : x_train.shape
								}
					sess.run(optimizer, feed_dict = feed_dict)
					batch_no += 1
				print "Training:epoch {}, iter {}, cost {}".format(i, batch_no, sess.run(cost, feed_dict : {
																									X : input_images,
																									X_r : input_images_r,
																									input_dims : input_images.shape
																								}))

			save = self.__saver.save(sess, self.__save_dir)	
			print "Model is saved in  file: ",save


	def softmax(self, inputs):

		input_dim = inputs.get_shape().as_list()
		num_dim = len(input_dim)

		return tf.nn.softmax(
					tf.reshape(inputs,[-1, input_dim[-1]])
				)

	def test(self, test_images, test_images_r):
		with tf.Session() as sess:
			if os.path.isfile(save_dir+"model.ckpt") == False:
				print "No trained model available"
			else:
				self.__saver.restore()

			output, cost = sess.run([output_prob, cost], feed_dict : {
												X : input_images,
												X_r : input_images_r,
												input_dims : input_images.shape
												})
			output = self.__generate(sess, 8, 8, 25)
			plot_100_figure(output, '{}/generated_only_images.jpg'.format(self.out_dir))


	def __generate(self, sess, height, width, num):
		X_samples = floatX(numpy.zeros((num, height, width, 1)))
		out = numpy.zeros((num, height, width, 3))

		for i in range(height):
			for j in range(width):
				samples = sess.run(pred, feed_dict = {X : X_samples})
				X_samples[:,i,j,:] = samples[:,i,j,:]
				# X_samples[:,i,j,:] = downscale_images(samples[:,i,j,:], Q_LEVELS - 1)

		return X_samples

args = parser.parse_args()
main().train(args)

