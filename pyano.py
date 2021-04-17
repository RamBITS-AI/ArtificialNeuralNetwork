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


class Neuron:
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
		return self.layer.activation.activation(self.input, [neuron.weights for neuron in self.layer.neurons])


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
		

		# if self.prev_layer != None and (type(self) != OutputLayer or type(self.activation) != SoftMaxActivation):
		# 	for n, neuron in enumerate(self.neurons):
		# 		neuron.output = neuron.get_activation()
		# if self.next_layer != None:
		# 	for nn, next_layer_neuron in enumerate(self.next_layer.neurons):
		# 		next_layer_neuron.input = 0
		# 	for nn, next_layer_neuron in enumerate(self.next_layer.neurons):
		# 		for n, neuron in enumerate(self.neurons):
		# 			next_layer_neuron.input += neuron.output * neuron.weights[nn]
		# 	for nn, next_layer_neuron in enumerate(self.next_layer.neurons):
		# 		next_layer_neuron.input += self.bias
		# 		if (next_layer_neuron.input > 100 or next_layer_neuron.input < -100):
		# 			print("====ERROR====>", next_layer_neuron.input)
		# elif self.prev_layer != None and type(self.activation) == SoftMaxActivation:
		# 	smax = self.get_activation()
		# 	for i_max, smax_value in enumerate(smax):
		# 		self.neurons[i_max].output = smax_value


class ActivationLayer(NeuralLayer):
	def __init__(
		self) -> None:
		super().__init__()
		self.activation = None
	
class WeightedLayer(NeuralLayer):
	def __init__(
		self) -> None:
		super().__init__()
		self.neurons: [WeightedNeuron] = []
		self.bias = utils.generate_bias()
		self.prev_bias = 0
		self.partial_error_derivative_of_bias = 0


	def attachToNextLayer(self, n_inputs, layer: NeuralLayer):
		self.next_layer = layer
		layer.prev_layer = self
		for neuron in self.neurons:
			neuron.weights = utils.generate_weights(n_inputs + 1)
			neuron.next_weights = utils.generate_zeros(n_inputs + 1)
			neuron.partial_error_derivatives = utils.generate_zeros(n_inputs + 1)
			neuron.prev_weights = utils.generate_zeros(n_inputs + 1)
		bias = utils.generate_bias()
		if type(layer) == OutputLayer:
			output_layer = layer
			for on, output_neuron in enumerate(output_layer.neurons):
				output_neuron.weights = utils.generate_weights(len(self.neurons) + 1)
				output_neuron.next_weights = utils.generate_zeros(len(self.neurons) + 1)
				output_neuron.partial_error_derivatives = utils.generate_zeros(len(self.neurons) + 1)
				output_neuron.prev_weights = utils.generate_zeros(len(self.neurons) + 1)
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


	def backward_propagate(self, targets, learning_rate):
		hidden_layer = self.prev_layer

		for hn, hidden_neuron in enumerate(hidden_layer.neurons):
			hidden_neuron.partial_error_derivatives = utils.generate_zeros(len(self.neurons))

		for hn, hidden_neuron in enumerate(hidden_layer.neurons):
			for on, output_neuron in enumerate(self.neurons):
				if type(self.activation) is SoftMaxActivation:
					derivatives = output_neuron.get_derivative()
					if derivatives.shape != (6,):
						raise ValueError("Derivative is Multidimensional!")
					derivative = derivatives[on]
				hidden_neuron.partial_error_derivatives[on] += -1 * \
					(targets[on] - output_neuron.output) * \
					derivative * hidden_neuron.output
				hidden_neuron.next_weights[on] = hidden_neuron.weights[on] - (learning_rate * hidden_neuron.partial_error_derivatives[on])
		
		# hidden_neuron = hidden_layer.neurons[1]

		# for on, output_neuron in enumerate(self.neurons):
		# 	if type(self.activation) is SoftMaxActivation:
		# 		derivatives = output_neuron.get_derivative()
		# 		if derivatives.shape != (6,):
		# 			raise ValueError("Derivative is Multidimensional!")
		# 		derivative = derivatives[on]
		# 	hidden_neuron.partial_error_derivatives[on] += -1 * \
		# 		(targets[on] - output_neuron.output) * \
		# 		derivative * hidden_neuron.output
		# 	hidden_neuron.next_weights[on] = hidden_neuron.weights[on] - (learning_rate * hidden_neuron.partial_error_derivatives[on])
		
		# hidden_neuron = hidden_layer.neurons[2]

		# for on, output_neuron in enumerate(self.neurons):
		# 	if type(self.activation) is SoftMaxActivation:
		# 		derivatives = output_neuron.get_derivative()
		# 		if derivatives.shape != (6,):
		# 			raise ValueError("Derivative is Multidimensional!")
		# 		derivative = derivatives[on]
		# 	hidden_neuron.partial_error_derivatives[on] += -1 * \
		# 		(targets[on] - output_neuron.output) * \
		# 		derivative * hidden_neuron.output
		# 	hidden_neuron.next_weights[on] = hidden_neuron.weights[on] - (learning_rate * hidden_neuron.partial_error_derivatives[on])
		
		# hidden_neuron = hidden_layer.neurons[3]

		# for on, output_neuron in enumerate(self.neurons):
		# 	if type(self.activation) is SoftMaxActivation:
		# 		derivatives = output_neuron.get_derivative()
		# 		if derivatives.shape != (6,):
		# 			raise ValueError("Derivative is Multidimensional!")
		# 		derivative = derivatives[on]
		# 	hidden_neuron.partial_error_derivatives[on] += -1 * \
		# 		(targets[on] - output_neuron.output) * \
		# 		derivative * hidden_neuron.output
		# 	hidden_neuron.next_weights[on] = hidden_neuron.weights[on] - (learning_rate * hidden_neuron.partial_error_derivatives[on])
		

		# for on, output_neuron in enumerate(self.neurons):
		# 	for hn, hidden_neuron in enumerate(hidden_layer.neurons):
		# 		hidden_neuron.partial_error_derivatives[on] += -1 * \
		# 			(targets[on] - output_neuron.output) * \
		# 			output_neuron.get_derivative() * hidden_neuron.output
		# 	for hn, hidden_neuron in enumerate(hidden_layer.neurons):
		# 		hidden_neuron.next_weights[on] -= (learning_rate * hidden_neuron.partial_error_derivatives[on])

		# for on, output_neuron in enumerate(self.neurons):
		# hidden_layer.partial_error_derivative_of_bias -= (targets[on] - output_neuron.output)
		
		# hidden_layer.partial_error_derivative_of_bias /= len(self.neurons)
		# hidden_layer.partial_error_derivative_of_bias *= -1
		# hidden_layer.bias -= (learning_rate * hidden_layer.partial_error_derivative_of_bias)


	def full_backward_propagate(self, targets, learning_rate):
		self.backward_propagate(targets, learning_rate)

		prev_layer = self.prev_layer
		while (prev_layer.prev_layer != None):
			prev_layer.backward_propagate(self, targets, learning_rate)
			prev_layer = prev_layer.prev_layer

		l = 1
		prev_layer = self.prev_layer
		while (prev_layer != None):
			for pn, prev_neuron in enumerate(prev_layer.neurons):
				for w, weight in enumerate(prev_neuron.weights):
					prev_neuron.weights[w] = prev_neuron.next_weights[w]
					prev_neuron.next_weights[w] = 0
					# print(f"Weight @ Network[{l}][{pn}][{w}] = {prev_neuron.weights[w]}")
			prev_layer = prev_layer.prev_layer
			l -= 1

	def get_activation(self):
		W = [neuron.weights for neuron in self.prev_layer.neurons]
		x = [neuron.output for neuron in self.prev_layer.neurons]
		Z = np.dot(np.transpose(W), x)
		h = self.activation.activation(None, Z)
		return h
		


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


	def backward_propagate(self, output_layer, targets, learning_rate):
		for n_prev, neuron_prev in enumerate(self.prev_layer.neurons):
			neuron_prev.partial_error_derivatives = utils.generate_zeros(len(self.neurons))

		for n_prev, neuron_prev in enumerate(self.prev_layer.neurons):
			for n, neuron in enumerate(self.neurons):
				for n_next, neuron_next in enumerate(self.next_layer.neurons):
					derivative = neuron_next.get_derivative()
					if type(self.next_layer.activation) == SoftMaxActivation:
						derivative = derivative[n_next]
					neuron_prev.partial_error_derivatives[n] += \
						-1 * (targets[n_next] - neuron_next.output) * \
						derivative * \
						neuron.weights[n_next] * \
						neuron.get_derivative() * \
						neuron_prev.output

				neuron_prev.next_weights[n] = neuron_prev.weights[n] - (learning_rate * neuron_prev.partial_error_derivatives[n])

		# for on, output_neuron in enumerate(output_layer.neurons):
		# 	for n, neuron in enumerate(self.neurons):
		# 		for n_prev, neuron_prev in enumerate(self.prev_layer.neurons):
		# 			neuron_prev.partial_error_derivatives[n] += -1 * \
		# 				(targets[on] - output_neuron.output) * \
		# 				output_neuron.get_derivative() * neuron.weights[on] * \
		# 				neuron.get_derivative() * neuron_prev.output
		# 		for n_prev, neuron_prev in enumerate(self.prev_layer.neurons):
		# 			neuron_prev.next_weights[n] -= (learning_rate * neuron_prev.partial_error_derivatives[n])


		# input_layer.partial_error_derivative_of_bias += (targets[on] - output_neuron.output)
		# input_layer.bias -= (learning_rate * input_layer.bias * input_layer.partial_error_derivative_of_bias)
		# input_layer.partial_error_derivative_of_bias += -1 * (targets[on] - output_neuron.output)
		# input_layer.bias -= (learning_rate * input_layer.bias * input_layer.partial_error_derivative_of_bias)

		# self.prev_layer.partial_error_derivative_of_bias += self.partial_error_derivative_of_bias
		
		# hidden_layer.partial_error_derivative_of_bias /= len(self.neurons)
		# hidden_layer.partial_error_derivative_of_bias *= -1
		# self.prev_layer.bias -= (learning_rate * self.prev_layer.partial_error_derivative_of_bias)