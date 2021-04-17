import abc
import enum
import utils


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

class ActivationNeuron(Neuron):
	def __init__(
		self) -> None:
		super().__init__()

	def get_activation(self):
		return self.layer.activation.activation(self.input, [neuron.weights for neuron in self.layer.prev_layer.neurons])

	def get_derivative(self):
		return self.layer.activation.derivative(self.input, self.output, [neuron.output for neuron in self.layer.neurons])

    # # Determine how much the neuron's total input has to change to move closer to the expected output
    # #
    # # Now that we have the partial derivative of the error with respect to the output (∂E/∂yⱼ) and
    # # the derivative of the output with respect to the total net input (dyⱼ/dzⱼ) we can calculate
    # # the partial derivative of the error with respect to the total net input.
    # # This value is also known as the delta (δ) [1]
    # # δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
    # #
	# def calculate_pd_error_wrt_total_net_input(self, target_output):
	# 	return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input()

	# # The partial derivate of the error with respect to actual output then is calculated by:
	# # = 2 * 0.5 * (target output - actual output) ^ (2 - 1) * -1
	# # = -(target output - actual output)
	# #
	# # The Wikipedia article on backpropagation [1] simplifies to the following, but most other learning material does not [2]
	# # = actual output - target output
	# #
	# # Alternative, you can use (target - output), but then need to add it during backpropagation [3]
	# #
	# # Note that the actual output of the output neuron is often written as yⱼ and target output as tⱼ so:
	# # = ∂E/∂yⱼ = -(tⱼ - yⱼ)
	# def calculate_pd_error_wrt_output(self, target_output):
	# 	return -(target_output - self.output)

	# # The total net input into the neuron is squashed using logistic function to calculate the neuron's output:
	# # yⱼ = φ = 1 / (1 + e^(-zⱼ))
	# # Note that where ⱼ represents the output of the neurons in whatever layer we're looking at and ᵢ represents the layer below it
	# #
	# # The derivative (not partial derivative since there is only one variable) of the output then is:
	# # dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)
	# def calculate_pd_total_net_input_wrt_input(self):
	# 	return self.layer.activation.derivative(self.output) #self.output * (1 - self.output)

	# # The total net input is the weighted sum of all the inputs to the neuron and their respective weights:
	# # = zⱼ = netⱼ = x₁w₁ + x₂w₂ ...
	# #
	# # The partial derivative of the total net input with respective to a given weight (with everything else held constant) then is:
	# # = ∂zⱼ/∂wᵢ = some constant + 1 * xᵢw₁^(1-0) + some constant ... = xᵢ
	# def calculate_pd_total_net_input_wrt_weight(self):
	# 	return self.input


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
	
	def forward_propagate(self):
		if self.prev_layer != None:
			for n, neuron in enumerate(self.neurons):
				neuron.output = neuron.get_activation()
		if self.next_layer != None:
			for nn, next_layer_neuron in enumerate(self.next_layer.neurons):
				next_layer_neuron.input = 0
			for nn, next_layer_neuron in enumerate(self.next_layer.neurons):
				for n, neuron in enumerate(self.neurons):
					next_layer_neuron.input += neuron.output * neuron.weights[nn]
			for nn, next_layer_neuron in enumerate(self.next_layer.neurons):
				next_layer_neuron.input += self.bias


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
		self.bias = 1
		self.prev_bias = 0
		self.partial_error_derivatives = []

	def attachToNextLayer(self, layer: NeuralLayer):
		self.next_layer = layer
		layer.prev_layer = self
		for neuron in self.neurons:
			neuron.weights = utils.generate_weights(len(layer.neurons))
			neuron.next_weights = utils.generate_zeros(len(layer.neurons))
			neuron.partial_error_derivatives = utils.generate_zeros(len(layer.neurons))
			neuron.prev_weights = utils.generate_zeros(len(layer.neurons))
		bias = utils.generate_bias()
		return layer


class InputLayer(WeightedLayer):
	def __init__(
		self,
		n_inputs) -> None:
		super().__init__()
		self.neurons = [WeightedNeuron() for i in range(n_inputs)]
		for input_neuron in self.neurons:
			input_neuron.layer = self

	def full_forward_propagate(self, inputs):
		for n, neuron in enumerate(self.neurons):
			neuron.output = inputs[n]

		current_layer = self
		output_layer = None
		
		while current_layer != None:
			current_layer.forward_propagate()
			output_layer = current_layer
			current_layer = current_layer.next_layer

		return [output_neuron.output for output_neuron in output_layer.neurons]

		# Update network weights with error
	def update_weights(self, inputs, learning_rate, mu):
		layer = self
		while (layer != None):
			if type(layer) != InputLayer:
				inputs = [neuron.output for neuron in layer.prev_layer.neurons]
				for neuron in layer.neurons:
					for pn, prev_neuron in enumerate(layer.prev_layer.neurons):
						for i, input in enumerate(inputs):
							temp = learning_rate * neuron.delta * input + mu * prev_neuron.prev_weights[i]
							prev_neuron.weights[i] += temp
							prev_neuron.prev_weights[i] = temp
					temp = learning_rate * neuron.delta + mu * layer.prev_layer.prev_bias
					layer.prev_layer.bias += temp
					layer.prev_layer.prev_bias = temp
			layer = layer.next_layer


class OutputLayer(ActivationLayer):
	def __init__(
		self,
		n_distinct_outputs,
		activation) -> None:
		super().__init__()	
		self.neurons = [ActivationNeuron() for i in range(n_distinct_outputs)]
		for output_neuron in self.neurons:
			output_neuron.layer = self
		self.activation = activation

	# Backpropagate error and store in neurons
	def backward_propagate_error(self, targets, learning_rate):
		layer = self
		while (layer != None): # From Output Layer to Input Layer
			errors = list()
			if layer.next_layer != None: # If it's not the Output Layer
				# Hidden Layers and Input Layer
				for n, neuron in enumerate(layer.neurons): #TODO: REVISIT
					error = 0.0
					for nn, next_neuron in enumerate(layer.next_layer.neurons):
						error += (neuron.weights[nn] * next_neuron.delta) #TODO: REVISIT
					errors.append(error)
			else:
				# Output Layer
				errors = [(targets[on] - output_neuron.output) for on, output_neuron in enumerate(layer.neurons)]
			# All Layers
			if type(layer) is InputLayer:
				pass
				# input_layer = layer
				# for inn, input_neuron in enumerate(input_layer.neurons):
				# 	neuron.delta = errors[inn] * utils.tanh_derivative(input_neuron.output)
			else:
				for n, neuron in enumerate(layer.neurons):
					neuron.delta = errors[n] * neuron.get_derivative()
			layer = layer.prev_layer
	
				
	def backward_propagate(self, targets, learning_rate):
		for n_prev, neuron_prev in enumerate(self.prev_layer.neurons):
			neuron_prev.partial_error_derivatives = utils.generate_zeros(len(self.neurons))
		for on, output_neuron in enumerate(self.neurons):
			for n_prev, neuron_prev in enumerate(self.prev_layer.neurons):
				neuron_prev.partial_error_derivatives[on] += -1 * (targets[on] - output_neuron.output) * output_neuron.get_derivative() * neuron_prev.output
			for n_prev, neuron_prev in enumerate(self.prev_layer.neurons):
				neuron_prev.next_weights[on] -= (learning_rate * neuron_prev.partial_error_derivatives[on])

	def full_backward_propagate(self, targets, learning_rate):
		self.backward_propagate(targets, learning_rate)

		prev_layer = self.prev_layer
		while (prev_layer.prev_layer != None):
			prev_layer.backward_propagate(self, targets, learning_rate)
			prev_layer = prev_layer.prev_layer

		prev_layer = self.prev_layer
		while (prev_layer != None):
			for neuron in prev_layer.neurons:
				for w, weight in enumerate(neuron.weights):
					neuron.weights[w] = neuron.next_weights[w]
					neuron.next_weights[w] = 0
			prev_layer = prev_layer.prev_layer

		# # 1. Output neuron deltas
		# pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.neurons)
		# for o, output_neuron in enumerate(self.neurons):

		# 	# ∂E/∂zⱼ
		# 	pd_errors_wrt_output_neuron_total_net_input[o] = output_neuron.calculate_pd_error_wrt_total_net_input(targets[o])

		# # 2. Hidden neuron deltas
		# pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.prev_layer.neurons)
		# for h, hidden_neuron in enumerate(self.prev_layer.neurons):

		# 	# We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
		# 	# dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
		# 	d_error_wrt_hidden_neuron_output = 0
		# 	for o, output_neuron in enumerate(self.neurons):
		# 		d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * hidden_neuron.weights[h]

		# 	# ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
		# 	pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * hidden_neuron.calculate_pd_total_net_input_wrt_input()

		# # 3. Update output neuron weights
		# for o, output_neuron in enumerate(self.neurons):
		# 	for h, hidden_neuron in enumerate(self.prev_layer.neurons):

		# 		# ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
		# 		pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * output_neuron.input#.calculate_pd_total_net_input_wrt_weight()

		# 		# Δw = α * ∂Eⱼ/∂wᵢ
		# 		hidden_neuron.weights[h] -= learning_rate * pd_error_wrt_weight

		# # 4. Update hidden neuron weights
		# for h, hidden_neuron in enumerate(self.prev_layer.neurons):
		# 	for i, input_neuron in enumerate(self.prev_layer.prev_layer.neurons):
		# 		for iw, input_weight in enumerate(input_neuron.weights):

		# 			# ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
		# 			pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * hidden_neuron.input #.calculate_pd_total_net_input_wrt_weight()

		# 			# Δw = α * ∂Eⱼ/∂wᵢ
		# 			hidden_neuron.weights[h] -= learning_rate * pd_error_wrt_weight


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

	# def backward_propagate(self, output_layer, targets, learning_rate):
	# 	for n_prev, neuron_prev in enumerate(self.prev_layer.neurons):
	# 		neuron_prev.partial_error_derivatives = utils.generate_zeros(len(self.neurons))
	# 	for on, output_neuron in enumerate(output_layer.neurons):
	# 		for n, neuron in enumerate(self.neurons):
	# 			for n_prev, neuron_prev in enumerate(self.prev_layer.neurons):
	# 				neuron_prev.partial_error_derivatives[n] += -1 * (targets[on] - output_neuron.output) * \
	# 					output_neuron.get_derivative() * neuron.weights[on] * \
	# 					neuron.get_derivative() * neuron_prev.output
	# 			for n_prev, neuron_prev in enumerate(self.prev_layer.neurons):
	# 				neuron_prev.next_weights[n] -= (learning_rate * neuron_prev.partial_error_derivatives[n])