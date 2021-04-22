""" 
    layers.py
    
	Created on 19-April-2021 @ 08:29 PM IST

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
import pyano as dnn

import abc


class Network(abc.ABC):
	def __init__(
		self) -> None:
		super().__init__()


@Network.register
class Sequential:
	def __init__(
		self,
		input_layer: dnn.InputLayer) -> None:
		super().__init__()
		self.input_layer: dnn.InputLayer = input_layer
		self.layers: [dnn.NeuralLayer] = [input_layer]
		self.output_layer: dnn.OutputLayer = None


	def add(
		self,
		layer: dnn.NeuralLayer) -> None:
		self.layers.append(layer)
		prev_layer = self.layers[len(self.layers) - 2]
		prev_layer.attachToNextLayer(layer)
		if type(layer) is dnn.OutputLayer:
			self.output_layer = layer


	def forward_propagate(
		self,
		inputs) -> list:
		return self.input_layer.full_forward_propagate(inputs)


	def backward_propagate(
		self,
		inputs,
		targets,
		learning_rate,
		mu) -> None:

		for layer in reversed(self.layers):
			layer.backward_propagate_error(targets)

		for layer in self.layers:
			layer.update_weights(inputs, learning_rate, mu)


	def predict(self, row):
		outputs = self.forward_propagate(row)

		prev_final_output = 0
		final_output = 0
		final_outputs = []

		for neuron in self.output_layer.neurons:
			final_output = max(prev_final_output, neuron.output)
			prev_final_output = final_output
			final_outputs.append(neuron.output)

		n = final_outputs.index(final_output)

		print("Final Output:", n)

		return n