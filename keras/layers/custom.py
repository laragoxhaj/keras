from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class DAN2(Layer):
'''
Custom 4-node layer as per DAN2 specs

CAKE nodes are weighted sum of previous layer's CAKE, CURNOLE, and C nodes
	Final CAKE node represents dependent variable
CURNOLE nodes in the hidden layer and their respective weights are designed to capture as much of the remaining (nonlinear) part of the process as possible

Usage:
	model = Sequential()
	model.add(DAN2(1, input_dim=16))

Args:
	output_dim: int > 0
		Generally set to 1
	init: name of initialization fn for layer weights
		or alternatively, Theano fn to use for weights initialization.
		Relevant only if you don't pass a `weights` argument.
	weights: list of numpy arrays to set as initial weights.
		List should have 2 elements, of shape `(input dim, output dim)`
		and (output_dim,) for weights and biases respectively
	input_dim: dimensionality of the input (int)
		Required when using this layer as the first in the model

	# Input shape: 2D tensor w/ shape `(nb_samples, input_dim)`
	# Output shape: 2D tensor w/ shape `(nb_samples, output_dim)`

'''
	def __init__(self, output_dim=1, init='glorot_uniform', weights=None, input_dim=None, **kwargs)
		# TODO: check default init / implement?
		self.init = initializations.get(init)
		self.output_dim = output_dim
		self.input_dim = input_dim

		self.initial_weights = weights

		if self.input_dim:
			kwargs['input_shape'] = (self.input_dim,)
		super(Dan2, self).__init__(**kwargs)

	def build(self, input_shape):
		'''
		Define weights
		'''
		# TODO: implement logic for all weights
		assert len(input_shape) == 2
		input_dim = input_shape[1]
		self.input_spec = [InputSpec(dtype=K.floatx(),
						shape=(None, input_dim))]
		self.W = self.init((1, 5),
					name='{}_b'.format(self.name))
	
		self.trainable_weights = [self.W]

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

		#initial_weight_value = np.random.random((input_dim, output_dim))
		#self.W = K.variable(initial_weight_value)

	def call(self, x, mask=None):
		'''
		Define logic of layer
		'''
		# TODO: implement logic for all nodes
		#TODO: want to support masking?

		R = np.ones((input_dim,), dtype=np.int)
		RXn = (K.dot(R, x)) / (sqrt(K.dot(R, R))*sqrt(K.dot(x, x)))
		aRX = acos(RXn)
		u = self.W(0, 0)
		abcd = self.W(0, 1:end)

		C = 1		# constant node
		F = x		# CAKE node
		G = cos(u*aRX)	# CURNOLE node
		H = sin(u*aRX)
		return K.dot([C, F, G, H], abcd)

	def get_output_shape_for(self, input_shape):
		'''
		Define shape transformation logic in case layer modifies shape of input
		Allows Keras to do automatic shape inference
		'''
		assert input_shape and len(input_shape) == 2
		return (input_shape[0], self.output_dim)

	def get_config(self):
		config = {'output_dim': self.output_dim,
			'init': self.init.__name__,
			'activation': 'DAN2 Fourier'
			'input_dim': self.input_dim}
		base_config = super(Dense, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
