from typing import Callable, List, Sequence, Tuple, cast

import jax
from jax import Array, numpy as jnp
import numpy as np


type Array1d = Array
type Array2d = Array
rng = np.random.default_rng()


class Layer:
	# the shape dictates the number of outputs and the number of inputs this Layer processes
	shape: Tuple[int, int]
	# specify if this layer is the last layer in the neural network (the output layer).
	# if it is the output layer, then the non-linear activation function will not be applied in the forwward direction.
	is_output: bool
	# the weights matrix, with the dimensions `shape`
	weights: Array2d
	# the bias matrix, with the dimensions `(shape[0], 1)`
	biases: Array1d
	# the non-linear activation functions
	# activations: List[Callable[[float], float]] # for now, I will just use a ReLU on all output variables to speed it up
	activation: Callable[[Array1d], Array1d]

	def __init__(self, shape: Tuple[int, int], is_output=False) -> None:
		(outputs, inputs) = shape
		self.shape = (outputs, inputs)
		self.is_output = is_output
		self.weights = self.init_weights(outputs, inputs)
		self.biases = self.init_biases(outputs)
		self.activation = self.init_activations(outputs)

	def forward(self, inputs: Array1d) -> Array1d:
		"""compute the layer's output in the forward direction.

		\\vec{outputs} = activation_fn(\\vec{biases} + weights \\cdot \\vec{inputs})
		"""
		linear_vector_output = jnp.dot(self.weights, inputs) + self.biases
		if self.is_output:
			return linear_vector_output
		return self.activation(linear_vector_output)

	def __call__(self, inputs: Array1d) -> Array1d:
		return self.forward(inputs)

	@staticmethod
	def init_weights(outputs: int, inputs: int) -> Array2d:
		"""initialize the weights matrix of the layer, with random normal values (mean = 0, std = 1.0),
		scaled down by a normalization factor that equals to the number of elements in the matrix.

		:param outputs: number of outputs variables this layer produces (number of matrix rows)
		:param inputs: number of input variables this layer takes in (number of matrix columns)
		:return: the weights matrix of this layer
		:rtype: Array2d[float32]
		"""
		normalization_factor = 1.0 / (outputs * inputs * 1.0)
		return jnp.array(normalization_factor * rng.standard_normal((outputs, inputs,), jnp.float32))

	@staticmethod
	def init_biases(outputs: int) -> Array1d:
		"""initialize the biases vector (column-matrix) of the layer, with zero initial values.

		:param outputs: number of outputs variables this layer produces (number of matrix rows)
		:return: the vector (column-matrix) of biases of this layer
		:rtype: Array1d[float32]
		"""
		return jnp.zeros((outputs,), jnp.float32)

	@staticmethod
	def init_activations(outputs: int) -> Callable[[Array1d], Array1d]:
		# TODO: make it per-node customizable. i.e.: `List[Callable[[float], float]]`
		return jax.nn.relu


class NeuralNetwork:
	# dictates the number of inputs this NeuralNetwork takes in
	inputs: int
	# the shape dictates the number of outputs and the number of inputs this NeuralNetwork processes. (i.e. `(outputs_len, inputs_len)`)
	# you should at least have added one layer for this to change, otherwise it will behave similar to outputs = inputs.
	# (i.e. `self.shape = (self.inputs, self.inputs)`)
	shape: Tuple[int, int]
	# all instances of hidden layers within this NeuralNetwork
	layers: List[Layer]

	def __init__(self, inputs: int) -> None:
		"""create an empty neural network.
		you should call the `appendNewLayer` method to add new layers.
		the last added layer is always thought of as the output layer, and so no activation function is applied on to it.
		the `shape` attribute of this neural network reflects the tuple value `(outputs, inputs)`, where:
		- the `outputs` is the number of outputs this neural network produces, derived dynamically from the output layer's (last layer's) output size.
		- the `inputs` is the number of inputs this neural network takes in, derived from the initial constant `input` argument that you provide when
		  creating an instance of this neural network. you cannot change the number of inputs after instantiating a neural network.

		:param inputs: the constant number of inputs (i.e. vector length) this neural network consumes
		"""
		self.inputs = inputs
		self.shape = (inputs, inputs)
		self.layers = []

	def appendLayer(self, layer: Layer) -> None:
		"""append an existing `Layer` (neural network layer) to this `NeuralNetwork`, and set it to being the output layer.
		in addition, the previous layer will drop its output layer status, thereby becoming non-linear.
		furthermore, the `shape` attribute of your `NeuralNetwork` instance will be updated to reflect the new output size.

		:param layer: the `Layer` to append
		"""
		layers = self.layers
		prev_layer = layers[-1] if len(layers) > 0 else None
		if prev_layer is not None:
			prev_layer.is_output = False
		layer.is_output = True
		layers.append(layer)
		self.shape = (layer.shape[0], self.inputs)

	def appendNewLayer(self, outputs: int) -> None:
		"""append a new layer to this `NeuralNetwork`, by simply providing your desired output size.

		:param layer: the `Layer` to append
		"""
		prev_layer_outputs = self.shape[0]
		new_layer = Layer((outputs, prev_layer_outputs))
		self.appendLayer(new_layer)

	def forward(self, inputs: Array1d) -> Array1d:
		"""compute the neural network's prediction for output in the forward direction. """
		for layer in self.layers:
			inputs = layer.forward(inputs)
		return inputs

	def __call__(self, inputs: Array1d) -> Array1d:
		return self.forward(inputs)

	def loss(self, input_data: Sequence[Array1d] | Array2d, output_data: Sequence[Array1d] | Array2d) -> float:
		"""compute the loss (error) between your training data (`output_data` and `input_data`), and the network's predicted outputs.

		you are free to override the implementation of the loss function to suite your specific neural network's requirements, so long as the signature is the same.
		the `NeuralNetwork` base class simply returns the mean-squared-error (MSE) between your training `output_data`, and this neural network's prediction.

		:param input_data: a sequence of vectors corresponding to the input data.
		  each vector element in the `input_data` must have the same length as this NeuralNetwork's `inputs` attribute (or `shape[1]` attribute).
		:param output_data: a sequence of vectors corresponding to the output data.
		  each vector element in the `output_data` must have the same length as this NeuralNetwork's last layer's output length (or simply the `shape[0]` attribute).
		:return: the total positive residual (error) between the training output data (`output_data`), and this neural network's prediction.
		"""
		predicted_output_data_arr: Array2d = jnp.asarray([self.forward(input) for input in input_data])
		output_data_arr: Array2d = jnp.asarray(output_data)
		squared_difference_arr: Array2d = jnp.pow(predicted_output_data_arr - output_data_arr, 2)
		mse: float = cast(float, jnp.mean(squared_difference_arr, axis=None).item())
		return mse

# TODO: implement a TrainableNeuralNetwork next, that uses jax's autograd to differentiate NeuralNetwork.prototype.loss, and figure out the direction of the gradient descent.
