import tensorflow as tf
import numpy as np
from keras.datasets import cifar10
# def uniform(stdev, size):
#     """uniform distribution with the given stdev and size"""
# 	return numpy.random.uniform(
# 		low=-stdev * numpy.sqrt(3),
# 		high=stdev * numpy.sqrt(3),
# 		size=size
# 	).astype(theano.config.floatX)

# def get():
# 	pass


class Layer():
	"""docstring for Layer"""
	'''
	Generic Layer Template which all layers should inherit.
	Every layer should have a name and params attribute containing all
	trainable parameters for that layer.
	'''
	def __init__(self, name = ""):
		self.name = name
		# self.params = []

class WrapperLayer(Layer):
	def __init__(self, X, name=""):
		# self.params = []
		self.name = name
		self.X = X

	def output(self):
		return self.X

class Conv2D(Layer):
	"""
	Basic Convolution layer
	input : (batch_size, input_dim, height, width)
	"""
	def __init__(self, inp, input_channels, output_channels, filter_size, subsample = (1,1), border_mode='half', masktype = None, activation = tf.nn.relu, name = "", biases_initializer=tf.zeros_initializer):
		self.X = inp
		self.name = name
		self.subsample = subsample
		self.border_mode = border_mode
		self.filter_size = filter_size
		self.filter = self.getConvFilter(self.filter_size)
		output = tf.nn.conv2d(self.X, self.filter, strides = [1, 1, 1, 1], padding = "SAME")
		if biases_initializer != None:
			biases = tf.get_variable("biases", [output_channels,],tf.float32, biases_initializer)
			self.Y = tf.nn.bias_add(output, biases)
		else:
			self.Y = output
		if activation != None:
			self.Y = activation(self.Y)

	def getConvFilter(self, filter_size, masktype = None, name = ""):
		conv_filter = tf.Variable(tf.random_uniform(filter_size))

		if masktype != None:
			masktype = masktype.lower()
			mask = np.ones(filter_size, dtype = tf.float32)
			center_h = filter_size[0] // 2
			center_w = filter_size[1] // 2
			mask[center_h, center_w+1:, :, :] = 0.0
			mask[center_h +1 :, :, :, :] = 0.0
			if masktype == "a":
				mask[center_h, center_w, :, :] = 0.0
			return conv_filter*mask
		return conv_filter

			
	def output(self):
		self.Y


class pixelConv(object):
	"""
	
	input_shape: (batch_size, height, width, input_dim)

	"""
	def __init__(self, inp, input_dim = 3, feature_maps = 32, out_dim = , q_levels = None, num_layers = 6, activation=tf.nn.relu, name=""):

		if activation is None:
			apply_act = lambda r: r
		elif activation == 'relu':
			apply_act = tf.nn.relu
		elif activation == 'tanh':
			apply_act = tf.tanh
		else:
			raise Exception("{} activation not implemented!!".format(activation))

		self.X = tf.transpose(inp, perm = [0, 3, 1, 2])
		# return self.X
		'''first filter in the paper has 7x7 layers'''
		first_filter = 7

		v_stack = Conv2D(self.X, input_dim, feature_maps, [filter_size // 2, filter_size, input_dim, feature_maps])
		out_v = v_stack.get_output()

		v_plus_input = tf.concat(2, [out_v, self.X])
		
		h_stack = Conv2D(v_plus_input, input_dim + feature_maps, feature_maps, [1, filter_size // 2, input_dim + feature_maps, feature_maps], masktype = "a"])
		
		x_h = h_stack.get_output()
		x_v = out_v

		filter_size = 3

		for i in xrange(num_layers - 2):
			v_stack = Conv2D(x_v, feature_maps, feature_maps, [filter_size // 2, filter_size, feature_maps, feature_maps])
			v2h = Conv2D(v_stack.get_output(), feature_maps, feature_maps, [1, 1, feature_maps, feature_maps])
			out_v = v2h.get_output()
			v_plus_input = tf.concat(2, [out_v, x_h])
			h_stack = Conv2D(v_plus_input, 2*feature_maps, feature_maps, [filter_size // 2, filter_size, 2*feature_maps, feature_maps], activation = activation)
			h2h = Conv2D(h_stack.get_output(), feature_maps, feature_maps, [1, 1, feature_maps, feature_maps], activation = activation)
			x_v = out_v
			x_h = h2h.get_output() + x_h
		
		"""Combined layer"""
		combined_stack = Conv2D(x_h, feature_maps, feature_maps, [1, 1, feature_maps, feature_maps], border_mode = "valid")
		if q_levels != None:
			out_dim = input_dim
		else:
			out_dim = input_dim * q_levels

		combined_stack_final = Conv2D(combined_stack.get_output(), feature_maps, out_dim, [1, 1, feature_maps, out_dim], border_mode = "valid")

		pre_final_output = tf.traanspose(combined_stack_final.get_output(), perm = [0, 2, 3, 1])
		if q_levels:
			old_shape = pre_final_output.get_shape().as_list()
			self.Y = tf.reshape(pre_final_output, shape = [old_shape[0], old_shape[1], old_shape[2] // q_levels, -1])
		else:
			self.Y = pre_final_output




	def get_output(self):
		return self.Y


# (X_train_r, _), (X_test_r, _) = cifar10.load_data()
# X_train_r = X_train_r[:2500]
# print X_train_r.shape
# 		# X_train_r = upscale_images(downscale_images(X_train_r, 256), Q_LEVELS) 
# x = pixelConv(X_train_r, 3, 20).get_output()
# with tf.Session() as sess:
# 	sess.run(tf.global_variables_initializer())
# 	print sess.run(x).shape
class pixelConvGated(object):
	"""
	
	input_shape: (batch_size, height, width, input_dim)
	This class implements the gated version of the Pixel CNN

	"""
	def __init__(self, inp, input_dim = 3, feature_maps = 32, out_dim = , q_levels = None, num_layers = 6, activation=tf.nn.relu, name=""):

		if activation is None:
			apply_act = lambda r: r
		elif activation == 'relu':
			apply_act = tf.nn.relu
		elif activation == 'tanh':
			apply_act = tf.tanh
		else:
			raise Exception("{} activation not implemented!!".format(activation))
		self.X = inp
		# self.X = tf.transpose(inp, perm = [0, 3, 1, 2])
		# return self.X
		'''first filter in the paper has 7x7 layers'''
		first_filter = 7

		v_stack = Conv2D(self.X, input_dim, feature_maps, [filter_size // 2, filter_size, input_dim, feature_maps])
		out_v = v_stack.get_output()

		v_plus_input = tf.concat(2, [out_v, self.X])
		
		h_stack = Conv2D(v_plus_input, input_dim + feature_maps, feature_maps, [1, filter_size // 2, input_dim + feature_maps, feature_maps], masktype = "a"])
		
		x_h = h_stack.get_output()
		x_v = out_v

		filter_size = 3

		for i in xrange(num_layers - 2):
			v_stack = Conv2D(x_v, feature_maps, feature_maps, [filter_size // 2, filter_size, feature_maps, feature_maps])
			v2h = Conv2D(v_stack.get_output(), feature_maps, feature_maps, [1, 1, feature_maps, feature_maps])
			out_v = v2h.get_output()
			v_plus_input = tf.concat(2, [out_v, x_h])
			h_stack = Conv2D(v_plus_input, 2*feature_maps, feature_maps, [filter_size // 2, filter_size, 2*feature_maps, feature_maps], activation = activation)
			h2h = Conv2D(h_stack.get_output(), feature_maps, feature_maps, [1, 1, feature_maps, feature_maps], activation = activation)
			x_v = out_v
			x_h = h2h.get_output() + x_h
		
		"""Combined layer"""
		combined_stack = Conv2D(x_h, feature_maps, feature_maps, [1, 1, feature_maps, feature_maps], border_mode = "valid")
		if q_levels != None:
			out_dim = input_dim
		else:
			out_dim = input_dim * q_levels

		combined_stack_final = Conv2D(combined_stack.get_output(), feature_maps, out_dim, [1, 1, feature_maps, out_dim], border_mode = "valid")

		pre_final_output = tf.traanspose(combined_stack_final.get_output(), perm = [0, 2, 3, 1])
		if q_levels:
			old_shape = pre_final_output.get_shape().as_list()
			self.Y = tf.reshape(pre_final_output, shape = [old_shape[0], old_shape[1], old_shape[2] // q_levels, -1])
		else:
			self.Y = pre_final_output




	def get_output(self):
		return self.Y
			

