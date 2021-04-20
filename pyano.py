""" 
    pyano.py
    
	Created on 22-March-2021 @ 11:14 AM IST

    Created by Ramachandran Chandrasekaran

	Pyano - A Deep Neural Network Framework

	An deep artificial neural network that utilizes back propagation for training. 
	It has a total of 5 layers of neurons. The hidden layers have 4 each, 
	the input layer has 3 and the output layer has 6 neurons 
	(as per the number of distinct classes in the given dataset).

	Contributions:

	Snehanshu Saha Sir from BITS Pilani
		For Teaching the mathematical concepts and also showing with examples on how to develop a Neural Network.
		For guiding in many ways and sharing his wisdom in the art of machine learning.
		And for sharing a working Artificial MultiLayerPerceptron implementation in Python.
	Ramachandran C - M.Tech Student of Snehanshu Saha Sir in BITS Pilani and Primary Author of Pyano.
"""
import abc
import enum
import utils

import numpy as np 
 
class GradientDescents(enum.Enum):
    GradientDescent = 1,
    MiniBatchGradientDescent = 2,
    StochasticGradientDescent = 3


class Neuron(abc.ABC):
	def __init__(
		self) -> None:
		super().__init__()
		self.layer = None
		self.input = None
		self.output = None
		self.delta = None

class WeightedNeuron(Neuron):
	def __init__(
		self) -> None:
		super().__init__()
		self.weights = []
		self.next_weights = []
		self.prev_weights = []
		self.partial_error_derivatives = []


class ActivationNeuron(Neuron):
	def __init__(
		self) -> None:
		super().__init__()
		self.delta = 0


	def get_activation(self):
		return self.layer.activation.activation(self.input, self.weights)  # [neuron.weights for neuron in self.layer.neurons])


	def get_derivative(self):
		return self.layer.activation.derivative(self.input, self.output, [neuron.output for neuron in self.layer.neurons])


class HiddenNeuron(WeightedNeuron, ActivationNeuron):
	def __init__(
		self) -> None:
		super().__init__()


class Activation(abc.ABC):
	def __init__(
		self) -> None:
		super().__init__()
		self.activation = None
		self.derivative = None


@Activation.register
class TanhActivation():
	def __init__(
		self) -> None:
		super().__init__()
		self.activation = utils.tanh
		self.derivative = utils.tanh_derivative


@Activation.register
class SigmoidActivation():
	def __init__(
		self) -> None:
		super().__init__()
		self.activation = utils.sigmoid
		self.derivative = utils.sigmoid_derivative


@Activation.register
class SBAFActivation():
	def __init__(
		self) -> None:
		super().__init__()
		self.activation = utils.sbaf
		self.derivative = utils.sbaf_derivative


@Activation.register
class SoftMaxActivation():
	def __init__(
		self) -> None:
		super().__init__()
		self.activation = utils.softmax
		self.derivative = utils.softmax_derivative


class NeuralLayer(abc.ABC):
	def __init__(
		self) -> None:
		super().__init__()
		self.neurons = [Neuron]
		self.next_layer = None
		self.prev_layer = None


	# Calculate neuron activation for an input
	def activate(self, weights, inputs, bias):
		activation = 0
		for i in range(len(inputs)):
			if inputs[i] != None:
				activation += weights[i] * inputs[i]
		activation += bias
		return activation


	def forward_propagate(self, inputs):
		new_inputs = []
		for n, neuron in enumerate(self.neurons):
			neuron.input = self.activate(neuron.weights, inputs, self.bias)
			neuron.output = neuron.get_activation()
			new_inputs.append(neuron.output)
		return new_inputs


class ActivationLayer(NeuralLayer):
	def __init__(
		self) -> None:
		super().__init__()
		self.activation = None
	

	def get_activation(self):
		W = [neuron.weights for neuron in self.prev_layer.neurons]
		x = [neuron.output for neuron in self.prev_layer.neurons]
		Z = np.dot(np.transpose(W), x)
		h = self.activation.activation(None, Z)
		return h


class WeightedLayer(NeuralLayer):
	def __init__(
		self) -> None:
		super().__init__()
		self.neurons: [WeightedNeuron] = []
		self.bias = utils.generate_bias()
		self.prev_bias = 0
		self.partial_error_derivative_of_bias = 0


	def attachToNextLayer(self, layer: NeuralLayer):
		self.next_layer = layer
		layer.prev_layer = self
		n_neurons_in_next_layer = len(self.next_layer.neurons)
		for neuron in self.neurons:
			neuron.weights = utils.generate_weights(n_neurons_in_next_layer)
			neuron.next_weights = utils.generate_zeros(n_neurons_in_next_layer)
			neuron.partial_error_derivatives = utils.generate_zeros(n_neurons_in_next_layer)
			neuron.prev_weights = utils.generate_zeros(n_neurons_in_next_layer)
		bias = utils.generate_bias()
		if type(layer) == OutputLayer:
			output_layer = layer
			n_neurons_in_output_layer = len(self.neurons)
			for on, output_neuron in enumerate(output_layer.neurons):
				output_neuron.weights = utils.generate_weights(n_neurons_in_output_layer)
				output_neuron.next_weights = utils.generate_zeros(n_neurons_in_output_layer)
				output_neuron.partial_error_derivatives = utils.generate_zeros(n_neurons_in_output_layer)
				output_neuron.prev_weights = utils.generate_zeros(n_neurons_in_output_layer)
			output_neuron.bias = utils.generate_bias()
		return layer


class InputLayer(WeightedLayer, ActivationLayer):
	def __init__(
		self,
		n_inputs,
		activation) -> None:
		super().__init__()
		self.neurons = [HiddenNeuron() for i in range(n_inputs)]
		self.activation = activation
		for input_neuron in self.neurons:
			input_neuron.layer = self


	def backward_propagate_error(self, targets):
		hidden_layer = self.next_layer
		input_layer = self
		
		errors = list()
		
		# As this is not Output Layer
		for inn, input_neuron in enumerate(input_layer.neurons):
			error = 0.0
			for hn, hidden_neuron in enumerate(hidden_layer.neurons):
				error += (hidden_neuron.weights[inn] * hidden_neuron.delta)
			errors.append(error)

		for inn, input_neuron in enumerate(input_layer.neurons):
			input_neuron.delta = errors[inn] * input_neuron.get_derivative()


	def update_weights(self, inputs, learning_rate, mu):
		hidden_layer = self.next_layer
		input_layer = self

		for inn, input_neuron in enumerate(input_layer.neurons):
			for i in range(len(inputs)):
				temp = learning_rate * input_neuron.delta * inputs[i] + mu * input_neuron.prev_weights[i]
		
				input_neuron.weights[i] += temp

				input_neuron.prev_weights[i] = temp
			
			temp = learning_rate * input_neuron.delta + mu * input_layer.prev_bias
			input_layer.bias += temp
			input_layer.prev_bias = temp


	def full_forward_propagate(self, inputs):
		for n, neuron in enumerate(self.neurons):
			neuron.output = inputs[n]

		current_layer = self
		
		while current_layer != None:
			inputs = current_layer.forward_propagate(inputs)
			current_layer = current_layer.next_layer

		return inputs


class OutputLayer(WeightedLayer, ActivationLayer):
	def __init__(
		self,
		n_distinct_outputs,
		activation) -> None:
		super().__init__()	
		self.neurons = [HiddenNeuron() for i in range(n_distinct_outputs)]
		for output_neuron in self.neurons:
			output_neuron.layer = self
		self.activation = activation
	
				
	def backward_propagate_error(self, targets):
		output_layer = self
		hidden_layer = self.prev_layer
		
		errors = list()
		
		for on, output_neuron in enumerate(output_layer.neurons):
			errors.append(targets[on] - output_neuron.output)

		for on, output_neuron in enumerate(output_layer.neurons):
			derivative = output_neuron.get_derivative()
			if type(self.activation) is SoftMaxActivation:
				derivatives = derivative
				if derivatives.shape != (6,):
					raise ValueError("Derivative is Multidimensional!")
				derivative = derivatives[on]

			output_neuron.delta = errors[on] * derivative
		

	def update_weights(self, inputs, learning_rate, mu):
		output_layer = self
		hidden_layer = self.prev_layer

		# As this is not Input Layer
		inputs = [neuron.output for neuron in hidden_layer.neurons]

		for on, output_neuron in enumerate(output_layer.neurons):
			for i in range(len(inputs)):
				temp = learning_rate * output_neuron.delta * inputs[i] + mu * output_neuron.prev_weights[i]
		
				output_neuron.weights[i] += temp

				output_neuron.prev_weights[i] = temp
			
			temp = learning_rate * output_neuron.delta + mu * output_layer.prev_bias
			output_layer.bias += temp
			output_layer.prev_bias = temp


class HiddenLayer(WeightedLayer, ActivationLayer):
	def __init__(
		self,
		n_neurons,
		activation) -> None:
		super().__init__()
		self.neurons = [HiddenNeuron() for i in range(n_neurons)]
		for hidden_neuron in self.neurons:
			hidden_neuron.layer = self
		self.activation = activation


	def backward_propagate_error(self, targets):
		output_layer = self.next_layer
		hidden_layer = self
		input_layer = self.prev_layer
		
		errors = list()
		
		# As this is not Output Layer
		for hn, hidden_neuron in enumerate(hidden_layer.neurons):
			error = 0.0
			for on, output_neuron in enumerate(output_layer.neurons):
				error += (output_neuron.weights[hn] * output_neuron.delta)
			errors.append(error)

		for hn, hidden_neuron in enumerate(hidden_layer.neurons):
			hidden_neuron.delta = errors[hn] * hidden_neuron.get_derivative()
		

	def update_weights(self, inputs, learning_rate, mu):
		output_layer = self.next_layer
		hidden_layer = self
		input_layer = self.prev_layer

		# As this is not Input Layer
		inputs = [neuron.output for neuron in input_layer.neurons]

		for hn, hidden_neuron in enumerate(hidden_layer.neurons):
			for i in range(len(inputs)):
				temp = learning_rate * hidden_neuron.delta * inputs[i] + mu * hidden_neuron.prev_weights[i]
		
				hidden_neuron.weights[i] += temp

				hidden_neuron.prev_weights[i] = temp
			
			temp = learning_rate * hidden_neuron.delta + mu * hidden_layer.prev_bias
			hidden_layer.bias += temp
			hidden_layer.prev_bias = temp


