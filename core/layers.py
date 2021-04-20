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