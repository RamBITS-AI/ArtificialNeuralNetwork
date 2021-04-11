# Backprop on the Vowel Dataset
from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import enum
import pandas as pd
import numpy as np
import csv

class GradientDescents(enum.Enum):
    GradientDescent = 1,
    MiniBatchGradientDescent = 2,
    StochasticGradientDescent = 3

# Load a CSV file
def load_csv(filename):
    lines = csv.reader(open(filename, 'r'))
    dataset = list(lines)
    pd_ds = pd.read_csv(filename)
    return dataset, pd_ds


def ds_to_float(dataset):
    col_count = len(dataset[0])
    for col in range(col_count):
        column_to_float(dataset, col)
    

# Print Dataset
def print_dataset(dataset):
    row_count = len(dataset)
    for row in range(row_count):
        print("DATA {}".format(dataset[row]))


# Find the min and max values for each column
def min_max(dataset):
    zipped_ds = zip(*dataset)
    minmax = [{'min': min(column), 'max': max(column)} for column in zipped_ds]
    return minmax
 

# Rescale dataset columns to the range 0-1
def normalize(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-1):
            min = minmax[i]['min']
            max = minmax[i]['max']
            row[i] = (row[i] - min) / (max - min)
 

# Convert string column to float
def column_to_float(dataset, column):
    for row in dataset:
        try:
            row[column] = float(row[column])
        except ValueError:
            print("Error with row",column,":",row[column])
            pass
 

# Convert string column to integer
def column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = len(dataset) / n_folds
    fold_size = int(fold_size)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split
 

# Calculate accuracy percentage
def accuracy_met(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Print a Formatted Matrix
def print_matrix(matrix, padding = 4):
    print('\n'.join([''.join([f'{item:{padding}}' for item in row]) for row in matrix]))
 
 
# Print a Formatted Matrix
def print_network(network, padding = 4):
    print('\n\nLayer:\n\n'.join(['\n'.join([f'Weights => {neuron["weights"]} \n Prev => {neuron["prev"]}' for neuron in layer]) for i, layer in enumerate(network) if i < len(network) - 1]))
    print("\n\n\n")
 
def print_outputs_of_neurons(network, padding = 4):
    nn = network
    output_layer = nn.pop(len(nn)-1)
    input_layer = nn.pop(0)
    # print('\n\nLayer:\n\n'.join(['\n'.join([f'Output = {neuron['output']}' for neuron in layer]) for layer in nn]))

    # print('\n\nFinal Layer:\n\n')
    # print('\n'.join([f'Output = {neuron['output']}' for neuron in output_layer]))
    # print('\n'.join([f'Final Output = {neuron['final_output']}' for neuron in output_layer]))
    # print('\n'.join([f'Final Error = {neuron["final_error"]}' for neuron in output_layer]))
    nn.append(output_layer)
    print("\n\n\n")
    print("Max of outputs of each layer:")
    l = 0
    for layer in nn:
        outputs = []
        for neuron in layer:
            outputs.append(neuron['output'])
        max1 = max(outputs)
        print(f'Max(Outputs) of Layer-{l+2}= {max1} and Index(Max) = {outputs.index(max1)}')
        l += 1
    final_outputs = []
    for final_neuron in nn[l-1]:
        final_outputs.append(final_neuron['output'])
    final_max = max(final_outputs)
    print(f'Max(Final Outputs) of Final Layer-{l+1}= {final_max} and Index(Final Max) = {final_outputs.index(final_max)}')

    nn.insert(0, input_layer)

# Initialize a network
def initialize_network(n_input_columns, n_hidden_neurons, n_hidden_layers, n_distinct_possible_outputs):
    network = list()
    input_layer = [{'weights':[random() for i in range(n_hidden_neurons + 1)], 'prev':[0 for i in range(n_hidden_neurons + 1)]} for i in range(n_input_columns)]
    network.append(input_layer)
    for h in range(n_hidden_layers-1):
        hidden_layer = [{'weights':[random() for i in range(n_hidden_neurons + 1)], 'prev':[0 for i in range(n_hidden_neurons + 1)]} for i in range(n_hidden_neurons)]
        network.append(hidden_layer)
    hidden_layer = [{'weights':[random() for i in range(n_distinct_possible_outputs + 1)], 'prev':[0 for i in range(n_distinct_possible_outputs + 1)]} for i in range(n_hidden_neurons)]
    network.append(hidden_layer)
    output_layer = [{'prev':[0 for i in range(n_distinct_possible_outputs)]} for i in range(n_distinct_possible_outputs)]
    network.append(output_layer)
    print("The Layers in the network:")
    print_network(network)
    return network
 
# One Hot Coding
def one_hot_encoding(n_possible_outcomes, outcome):
    one_hot = [0 for i in range(n_possible_outcomes)]
    one_hot[outcome] = 1
    return one_hot


def print_delta_and_output(network):
    l = 0
    for layer in network:
        n = 0
        for neuron in layer:
            if 'delta' in neuron.keys():
                d = 0
                for delta in neuron['delta']:
                    print(f"network[{l}][{n}]['delta'][{d}] = {delta}")
                    d += 1
            if 'output' in neuron.keys():
                o = 0
                for output in neuron['delta']:
                    print(f"network[{l}][{n}]['output'][{o}] = {output}")
                    o += 1
            n += 1
        l += 1


def train_network_using_gradient_descent(pd_ds, dataset, network, train, l_rate, n_epoch, n_outputs, mu):
    direction = 1
    # prev_max_final_error = 1
    for row in train:
        r = 0
        for epoch in range(n_epoch):
            # if epoch == 0:
            #     prev_max_final_error = 1
            # o,c = batchnorm_forward(pd_ds, dataset, 0.01, 0.02, 'train')
            outputs = forward_propagate(network, row)
            #print(network)
            expected = one_hot_encoding(n_outputs, row[-1])
            #print("expected row{}\n".format(expected))
            # dx, dgamma, dbeta = batchnorm_backward(o, c)
            backward_propagate_error(network, expected, row[:-1], l_rate)
            # print("The Layers in the network after back prop:")
            # print_outputs_of_neurons(network)
            # print_network(network)
            # print_delta_and_output(network)


            # update_weights(network, row, l_rate, mu, direction)
            
            
            # print_delta_and_output(network)
            # print("\n\nThe Layers in the network after updating weights:\n\n")
            # print_outputs_of_neurons(network)
            # if r == len(train) - 1:


            final_errors = get_final_errors(network)
            
            
            # max_final_error = max(final_errors)
            # if max_final_error > prev_max_final_error:
            #     direction *= -1
            # prev_max_final_error = max_final_error
            # print_outputs_of_neurons(network)
            # print(f"Epoch = {epoch}")
            r += 1
            # print(expected)
            # for input_neuron in network[0]:
            #     print([weight for weight in input_neuron['weights']])
            # print('')


 # Train a network for a fixed number of epochs
def train_network_using_mini_batch_gradient_descent(pd_ds, dataset, network, train, l_rate, n_epoch, n_outputs, mu):
    direction = 1
    # prev_max_final_error = 1
    for epoch in range(n_epoch):
        r = 0
        for row in train:
            # if epoch == 0:
            #     prev_max_final_error = 1
            # o,c = batchnorm_forward(pd_ds, dataset, 0.01, 0.02, 'train')
            outputs = forward_propagate(network, row)
            #print(network)
            expected = one_hot_encoding(n_outputs, row[-1])
            #print("expected row{}\n".format(expected))
            # dx, dgamma, dbeta = batchnorm_backward(o, c)
            backward_propagate_error(network, expected, row[:-1], l_rate)
            # print("The Layers in the network after back prop:")
            # print_outputs_of_neurons(network)
            # print_network(network)
            # print_delta_and_output(network)
            
            
            # update_weights(network, row, l_rate, mu, direction)
            
            
            # print_delta_and_output(network)
            # print("\n\nThe Layers in the network after updating weights:\n\n")
            # print_outputs_of_neurons(network)
            # if r == len(train) - 1:
            
            
            # final_errors = get_final_errors(network)
            
            
            # max_final_error = max(final_errors)
            # if max_final_error > prev_max_final_error:
            #     direction *= -1
            # prev_max_final_error = max_final_error
            # print_outputs_of_neurons(network)
            # print(f"Epoch = {epoch}")
            r += 1
            # print(expected)
            # for input_neuron in network[0]:
            #     print([weight for weight in input_neuron['weights']])
            # print('')


def train_network_using_stochastic_gradient_descent(pd_ds, dataset, network, train, l_rate, n_epoch, n_outputs, mu):
    direction = 1
    # prev_max_final_error = 1
    for row in train:
        r = 0
        for epoch in range(n_epoch):
            # if epoch == 0:
            #     prev_max_final_error = 1
            # o,c = batchnorm_forward(pd_ds, dataset, 0.01, 0.02, 'train')
            outputs = forward_propagate(network, row)
            #print(network)
            expected = one_hot_encoding(n_outputs, row[-1])
            #print("expected row{}\n".format(expected))
            # dx, dgamma, dbeta = batchnorm_backward(o, c)
            backward_propagate_error(network, expected, row[:-1], l_rate)
            # print("The Layers in the network after back prop:")
            # print_outputs_of_neurons(network)
            # print_network(network)
            # print_delta_and_output(network)
            
            
            # update_weights(network, row, l_rate, mu, direction)
            
            
            # print_delta_and_output(network)
            # print("\n\nThe Layers in the network after updating weights:\n\n")
            # print_outputs_of_neurons(network)
            # if r == len(train) - 1:
            
            
            final_errors = get_final_errors(network)
            
            
            # max_final_error = max(final_errors)
            # if max_final_error > prev_max_final_error:
            #     direction *= -1
            # prev_max_final_error = max_final_error
            # print_outputs_of_neurons(network)
            # print(f"Epoch = {epoch}")
            r += 1
            # print(expected)
            # for input_neuron in network[0]:
            #     print([weight for weight in input_neuron['weights']])
            # print('')


# Calculate neuron activation for an input
def activate(i, weights, inputs, current_activation, act_strs):
    if i == 0: # if the first neuron in the layer
        bias = weights[-1]
    else:
        bias = 0
    activation = current_activation
    if i < (len(inputs) - 1):
        for j in range(len(weights)-1):
            if j < len(activation):
                activation[j] += weights[j] * inputs[i]
                act_strs[j] += f" + (weights[{j}] * inputs[{i}])"
            else:
                activation.append(weights[j] * inputs[i])
                act_strs.append(f"(weights[{j}] * inputs[{i}])")
    k = 0
    for act_node in activation:
        if bias != 0:
            act_node += bias
            act_strs[k] += f" + bias"
        k += 1
    return activation, act_strs
 

def sigmoid(activation):
    return 1.0 / (1.0 + exp(-activation))


def tanh(activation):
	return (exp(activation) - exp(-activation)) / (exp(activation) + exp(-activation))


def softmax(activation):
    e = 0
    act_nodes = []
    activations = []
    for act_node in activation:
        e = exp(act_node)
        act_nodes.append(e)

    sum_of_act_nodes = sum(act_nodes)
    for act_node in act_nodes:
        activations.append(act_node/sum_of_act_nodes)

    return activations


# # define data
# data = [1, 3, 2]
# # convert list of numbers to a list of probabilities
# result, act_nodes = softmax(data)
# # report the probabilities
# print(result)
# # report the sum of the probabilities
# print(sum(act_nodes))

def sigmoid_activation(current_activation):
    activation = []
    for act_node in current_activation:
        act_node = sigmoid(act_node)
        activation.append(act_node)
    return activation


def tanh_activation(current_activation):
    activation = []
    for act_node in current_activation:
        act_node = tanh(act_node)
        activation.append(act_node)
    return activation


def softmax_activation(current_activation):
    activation = softmax(current_activation)
    return activation


def batchnorm_forward(pd_ds, dataset, gamma, beta, mode='train', eps=1e-5,momentum=0.9, running_mean=None,running_var=None):
    N, D = pd_ds.shape
    if running_mean == None:
        running_mean=np.zeros(D, dtype=pd_ds.values.dtype)
    if running_var == None:
        running_var=np.zeros(D, dtype=pd_ds.values.dtype)

    sample_mean = pd_ds.mean(axis=0)
    sample_var = pd_ds.var(axis=0)
    std = np.sqrt(sample_var + eps)
    x_centered = pd_ds - sample_mean
    if mode == 'train':
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        
        x_norm = x_centered / std
        out = gamma * x_norm + beta
        
        cache = (x_norm, x_centered, std, gamma)
    elif mode == 'test':
        x_norm = (pd_ds - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta
        cache = (x_norm, x_centered, std, gamma)
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    running_mean = running_mean
    running_var = running_var

    return out, cache


# def get_pre_activations(network, n_layer, inputs):
#     if (n_layer < (len(network) - 1)):
#         current_layer = network[n_layer]
#         next_hidden_layer = network[n_layer] # This should never point to the output layer
#         for n, neuron in enumerate(current_layer):
#             bias = 0
#             weights = neuron['weights']
#             if n == 0:
#                 bias = weights[len(weights) - 1]
#             current_pre_activation = 0
#             for n_next, neuron_next in enumerate(next_hidden_layer):
#                 current_pre_activation += weights[n_next] * 

# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    act_strs = []
    for i, layer in enumerate(network):
        # if len(act_strs) > 0:
        #     for act_str in act_strs:
        #         print(act_str)
        act_strs = []
        new_inputs = []
        # l = 0
        activation = []
        if i < (len(network) - 1):
            for k in range(len(layer)):
                neuron = layer[k]
                # print(f'Layer-{i}, Neuron-{k}:')
                activation, act_strs = activate(k, neuron['weights'], inputs, activation, act_strs)
                # for act_str in act_strs:
                #     print(act_str)
        else:
            activation = inputs
        
        for k in range(len(layer)):
            neuron = layer[k]
            pre_activations = activation
            if i == len(network) - 1:
                new_inputs = tanh_activation(activation) #TODO: REMOVE
                # new_inputs = softmax_activation(activation) #TODO: UNCOMMENT
            elif i == len(network) - 2:
                new_inputs = tanh_activation(activation)
            else:
                new_inputs = tanh_activation(activation)
            activation = new_inputs

            inputs = new_inputs
            post_activations = new_inputs
            if (k < (len(layer) - 1)) and len(activation) > 0:
                max1 = max(activation)
                min1 = min(activation)
                if max1 != 0:
                    if max1 > 1 or min1 < 0:
                        print(f"===========> ERROR: min={min1}, max={max1}")
                        # for m in range(len(activation)):
                        #     activation[m] = (activation[m] - min1) / (max1 - min1)
                    j = 0

                    if (i + 1) < len(network):
                        layer1 = network[i + 1]
                        for neuron in layer1:
                            neuron['input'] = pre_activations[j]
                            neuron['output'] = new_inputs[j]
                            j += 1
                    else:
                        layer1 = network[i]
                        for neuron in layer1:
                            neuron['input'] = pre_activations[j]
                            neuron['output'] = new_inputs[j]
                            j += 1
                            # new_inputs.append(neuron['output'])
                    # inputs = new_inputs
                else:
                    print("max1 is 0.")
    return inputs
 

# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)
 

def batchnorm_backward(dout, cache):
    N = 1
    x_norm, x_centered, std, gamma = cache
    
    dgamma = (dout * x_norm).sum(axis=0)
    dbeta = dout
    
    dx_norm = dout * gamma
    dx = 1/N / std * (N * dx_norm - 
                      dx_norm - 
                      x_norm * (dx_norm * x_norm).sum(axis=0))

    return dx, dgamma, dbeta


def tanh_derivative(actual_input):
    return 1 - pow((tanh(actual_input)), 2)

prev_layer_for_derivatives = None
prev_layer_copy_for_derivatives = None

def get_sum_of_prev_derivatives(prev_layers, n_neuron, n_weight, is_deep = False):
    global prev_layer_for_derivatives, prev_layer_copy_for_derivatives
    if len(prev_layers) == 0:
        sum_of_prev_derivatives = 0.0
    else:
        sum_of_prev_derivatives = 0.0
        prev_layers_copy = [layer_copy for layer_copy in prev_layers]
        while (len(prev_layers_copy) > 0):
            prev_layer_for_derivatives = prev_layer_copy_for_derivatives
            prev_layer_copy_for_derivatives = prev_layers_copy.pop(0)
            if is_deep:
                if 'weights' in prev_layer_copy_for_derivatives[n_neuron].keys():
                    sum_of_prev_derivatives += prev_layer_copy_for_derivatives[n_weight]['derivative'] + prev_layer_copy_for_derivatives[n_weight]['weights'][0] + get_sum_of_prev_derivatives(prev_layers_copy, n_neuron, n_weight, True)
                else:
                    sum_of_prev_derivatives += prev_layer_copy_for_derivatives[n_weight]['derivative'] + get_sum_of_prev_derivatives(prev_layers_copy, n_neuron, n_weight, True)
            else:
                sum_of_prev_derivatives += prev_layer_copy_for_derivatives[n_weight]['derivative'] + get_sum_of_prev_derivatives(prev_layers_copy, n_neuron, n_weight, True)
    return sum_of_prev_derivatives

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected, inputs, learning_rate):
    global prev_layer_for_derivatives, prev_layer_copy_for_derivatives
    prev_layers = []
    prev_layer = None
    output_layer = None
    for l, layer in enumerate(reversed(network)):
        ll = (len(network) - 1) - l
        n_layers = len(network)
        if ll < n_layers - 1 and ll > 0:
            # print("Hidden Layer")
            for n, neuron in enumerate(layer):
                actual_input = neuron['input']
                neuron['derivative'] = tanh_derivative(actual_input)
                for w, weight in enumerate(neuron['weights']):
                    if w < len(neuron['weights']) - 1:
                        prev_layer_for_derivatives = None
                        prev_layer_copy_for_derivatives = None
                        sum_of_prev_derivatives = get_sum_of_prev_derivatives(prev_layers, n, w)
                        # print('original_weight = ', neuron['weights'][w])
                        neuron['weights'][w] = get_new_weight(output_layer[w]['first_error_derivative'], sum_of_prev_derivatives, neuron['output'], learning_rate, neuron['weights'][w])
                        # print('updated_weight = ', neuron['weights'][w])
            prev_layers.append(layer)
        elif ll == 0:
            # print("Input Layer")
            for n, neuron in enumerate(layer):
                for w, weight in enumerate(neuron['weights']):
                    if w < len(neuron['weights']) - 1:
                        prev_layer_for_derivatives = None
                        prev_layer_copy_for_derivatives = None
                        sum_of_prev_derivatives = get_sum_of_prev_derivatives(prev_layers, n, w)
                        # print('original_weight = ', neuron['weights'][w])
                        neuron['weights'][w] = get_new_weight(output_layer[w]['first_error_derivative'], sum_of_prev_derivatives, inputs[n], learning_rate, neuron['weights'][w])
                        # print('updated_weight = ', neuron['weights'][w])
        else:
            # print("Output Layer")
            for n, neuron in enumerate(layer):
                actual_input = neuron['input']
                actual_output = neuron['output']
                desired_output = expected[n]
                neuron['derivative'] = tanh_derivative(actual_input)
                neuron['first_error_derivative'] = -1 * (desired_output - actual_output)
            output_layer = layer
            prev_layers.append(layer)
        prev_layer = layer

def backward_propagate_error_1(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        final_errors = list()
        if i != len(network)-1:
            # for j in range(len(layer)):
            next_layer = network[i + 1]
            for neuron_in_current_layer in layer:
                error = 0.0
                k = 0
                for neuron_in_next_layer in next_layer:
                    error += (neuron_in_current_layer['weights'][k] * neuron_in_next_layer['delta'][k])
                    k += 1
                errors.append(error)
        else:
            output_layer = layer
            error = 0.0
            n = len(output_layer)
            for k in range(n):
                output_neuron = output_layer[k]
                # error += (output_neuron['weights'][k] * output_neuron['final_delta'][k])
                # errors.append(error)
                final_error = pow(expected[k] - output_neuron['input'], 2)/2
                final_errors.append(final_error)
                output_neuron['final_error'] = final_error
        for l in range(len(layer)):
            # if i > 0: # if this is not the input layer
            neuron = layer[l]
            if 'delta' not in neuron.keys():
                neuron['delta'] = []
            if i == (len(network) - 1): # if this is the output layer
                output_neuron = neuron
                if 'final_delta' not in output_neuron.keys():
                    output_neuron['final_delta'] = []
                for out_neu_2 in layer:
                    output_neuron['final_delta'].append(final_errors[l] * transfer_derivative(out_neu_2['input']))
                output_layer = layer
                error = 0.0
                output_neuron_1 = neuron
                for m in range(len(output_layer)):
                    # kk = 0
                    # for neuron in network[i + 1]:
                    error += (output_neuron_1['weights'][m] * output_neuron_1['final_delta'][m])
                    errors.append(error)
                for out_neu_1 in layer:
                    output_neuron['delta'].append(errors[l] * transfer_derivative(out_neu_1['input']))
            else:
                for neu in network[i+1]:
                    neuron['delta'].append(errors[l] * transfer_derivative(neu['output']))
            # print(f"Error[{l}]={errors[l]}")
            # print(f"Output[{l}]={neuron['output']}")

def get_final_errors(network):
    return [neuron['final_error'] for neuron in [layer for i, layer in enumerate(network) if i == len(network) - 1][0]]


def get_final_outputs(network):
    return [neuron['final_output'] for neuron in [layer for i, layer in enumerate(network) if i == len(network) - 1][0]]


def get_new_weight(first_error_derivative, sum_of_prev_derivatives, output_of_current_neuron, learning_rate, current_weight):
    delta = first_error_derivative + sum_of_prev_derivatives + output_of_current_neuron
    return current_weight - (learning_rate * delta)

# Update network weights with error
def update_weights(network, row, l_rate, mu, direction):
    for i in range(len(network)):
        inputs = row[:-1]
        if i > 1 and i < (len(network) - 1):
            inputs = [neuron['output'] for neuron in network[i]]
        elif i == (len(network) - 1):
            inputs = [neuron['final_output'] for neuron in network[i]]
        if i > 0:
            for j in range(len(inputs)):
                nrn = 0
                lyr = None
                if i == len(network) - 1:
                    lyr = network[i]
                else:
                    lyr = network[i+1]
                for neuron in lyr:
                    temp = 0
                    if i == len(network) - 1:
                        temp = l_rate * neuron['final_delta'][nrn] * inputs[j] + mu * neuron['final_prev'][nrn] * direction
                        network[i][j]['final_prev'][nrn] = temp
                    else:
                        temp = l_rate * neuron['delta'][nrn] * inputs[j] + mu * neuron['prev'][nrn] * direction                
                    network[i][j]['weights'][nrn] += temp
                    network[i][j]['prev'][nrn] = temp
                    nrn += 1
            n = 0
            # Update the bias in the first neuron and empty the biases of the other neurons in the layer
            for neuron in network[i]:
                # for j in range(len(inputs)):
                #     temp = l_rate * neuron['delta'][j] * inputs[j] + mu * neuron['prev'][j]                    
                #     neuron['weights'][j] += temp
                #     neuron['prev'][j] = temp
                if n == 0:
                    temp = l_rate * neuron['delta'][-1] + mu * neuron['prev'][-1] * direction
                    neuron['weights'][-1] += temp
                    neuron['prev'][-1] = temp
                else:
                    neuron['weights'][-1] = 0
                    neuron['prev'][-1] = 0
                n += 1
        else:
            for j in range(len(inputs)):
                nrn = 0
                for neuron in network[i+1]:
                    temp = l_rate * neuron['delta'][nrn] * inputs[j] + mu * neuron['prev'][nrn] * direction
                    network[i][j]['weights'][nrn] += temp
                    network[i][j]['prev'][nrn] = temp
                    nrn += 1
            nx = 0
            for neuron in network[i]:
                # for j in range(len(inputs)):
                #     temp = l_rate * neuron['delta'][j] * inputs[j] + mu * neuron['prev'][j]                    
                #     neuron['weights'][j] += temp
                #     neuron['prev'][j] = temp
                if nx == 0:
                    temp = l_rate * neuron['delta'][-1] + mu * neuron['prev'][-1] * direction
                    neuron['weights'][-1] += temp
                    neuron['prev'][-1] = temp
                else:
                    neuron['weights'][-1] = 0
                    neuron['prev'][-1] = 0
                nx += 1


# Make a prediction with a network
def predict(pd_ds, dataset, network, row):
    # o,c = batchnorm_forward(pd_ds, dataset, 0.01, 0.02, 'test')
    outputs = forward_propagate(network, row)
    n = 0
    prev_final_output = 0
    final_output = 0
    final_outputs = []
    for neuron in network[len(network)-1]:
        final_output = max(prev_final_output, neuron['output'])
        prev_final_output = final_output
        final_outputs.append(neuron['output'])
    n = final_outputs.index(final_output)
    # print("Network Layers:")
    # print_network(network)
    # print("Outputs of Neurons:")
    # print_outputs_of_neurons(network)
    # print("Final Outputs:", final_outputs)
    print("Final Output:", n)
    # print(outputs.index(max(outputs)))
    return n#, o, c # outputs.index(max(outputs))
 

# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(gradientDescent, pd_ds, dataset, train, test, l_rate, n_epoch, n_hidden_neurons, n_hidden_layers, mu):
    n_input_columns = len(train[0]) - 1 # Number of columns in the training dataset minus the one column which contains the output
    distinct_outputs = set([row[-1] for row in train])
    n_distinct_outputs = len(distinct_outputs)
    network = initialize_network(n_input_columns, n_hidden_neurons, n_hidden_layers, n_distinct_outputs)
    if (gradientDescent == GradientDescents.GradientDescent):
        train_network_using_gradient_descent(pd_ds, dataset, network, train, l_rate, n_epoch, n_distinct_outputs, mu)
    elif (gradientDescent == GradientDescents.MiniBatchGradientDescent):
        train_network_using_mini_batch_gradient_descent(pd_ds, dataset, network, train, l_rate, n_epoch, n_distinct_outputs, mu)
    else:
        train_network_using_stochastic_gradient_descent(pd_ds, dataset, network, train, l_rate, n_epoch, n_distinct_outputs, mu)
    #print("network {}\n".format(network))
    predictions = list()
    for row in test:
        prediction = predict(pd_ds, dataset, network, row)
        # prediction, o, c = predict(pd_ds, dataset, network, row)
        # dx, dgamma, dbeta = batchnorm_backward(0, c)
        predictions.append(prediction)
    return(predictions)


# def ssr_gradient(x, y, b):
#     res = b[0] + b[1] * x - y
#     return res.mean(), (res * x).mean()  # .mean() is a method of np.ndarray


# def sgd(
#     gradient, x, y, n_vars=None, start=None, learn_rate=0.1,
#     decay_rate=0.0, batch_size=1, n_iter=50, tolerance=1e-06,
#     dtype="float64", random_state=None
# ):
#     # Checking if the gradient is callable
#     if not callable(gradient):
#         raise TypeError("'gradient' must be callable")

#     # Setting up the data type for NumPy arrays
#     dtype_ = np.dtype(dtype)

#     # Converting x and y to NumPy arrays
#     x, y = np.array(x, dtype=dtype_), np.array(y, dtype=dtype_)
#     n_obs = x.shape[0]

#     if n_obs != y.shape[0]:
#         raise ValueError("'x' and 'y' lengths do not match")
#     xy = np.c_[x.reshape(n_obs, -1), y.reshape(n_obs, 1)]

#     # Initializing the random number generator
#     seed = None if random_state is None else int(random_state)
#     rng = np.random.default_rng(seed=seed)

#     # Initializing the values of the variables
#     vector = (
#         rng.normal(size=int(n_vars)).astype(dtype_)
#         if start is None else
#         np.array(start, dtype=dtype_)
#     )

#     # Setting up and checking the learning rate
#     learn_rate = np.array(learn_rate, dtype=dtype_)
#     if np.any(learn_rate <= 0):
#         raise ValueError("'learn_rate' must be greater than zero")

#     # Setting up and checking the decay rate
#     decay_rate = np.array(decay_rate, dtype=dtype_)
#     if np.any(decay_rate < 0) or np.any(decay_rate > 1):
#         raise ValueError("'decay_rate' must be between zero and one")

#     # Setting up and checking the size of minibatches
#     batch_size = int(batch_size)
#     if not 0 < batch_size <= n_obs:
#         raise ValueError(
#             "'batch_size' must be greater than zero and less than "
#             "or equal to the number of observations"
#         )

#     # Setting up and checking the maximal number of iterations
#     n_iter = int(n_iter)
#     if n_iter <= 0:
#         raise ValueError("'n_iter' must be greater than zero")

#     # Setting up and checking the tolerance
#     tolerance = np.array(tolerance, dtype=dtype_)
#     if np.any(tolerance <= 0):
#         raise ValueError("'tolerance' must be greater than zero")

#     # Setting the difference to zero for the first iteration
#     diff = 0

#     # Performing the gradient descent loop
#     for _ in range(n_iter):
#         # Shuffle x and y
#         rng.shuffle(xy)

#         # Performing minibatch moves
#         for start in range(0, n_obs, batch_size):
#             stop = start + batch_size
#             x_batch, y_batch = xy[start:stop, :-1], xy[start:stop, -1:]

#             # Recalculating the difference
#             grad = np.array(gradient(x_batch, y_batch, vector), dtype_)
#             diff = decay_rate * diff - learn_rate * grad

#             # Checking if the absolute difference is small enough
#             if np.all(np.abs(diff) <= tolerance):
#                 break

#             # Updating the values of the variables
#             vector += diff

#     return vector if vector.shape else vector.item()

# x = [[0,1,2,3,4,5,6],[0,1,2,3,4,5,6],[0,1,2,3,4,5,6],[0,1,2,3,4,5,6],[0,1,2,3,4,5,6],[0,1,2,3,4,5,6],[0,1,2,3,4,5,6]]
# y = [1,2,3,4,5,6,7]
# vector = sgd(
#     ssr_gradient, x, y, n_vars=2, learn_rate=0.0001,
#     decay_rate=0.8, batch_size=3, n_iter=100_000, random_state=0
# )
# print(vector)

# Evaluate an algorithm using a cross validation split
def run_algorithm(gradientDescent, pd_ds, dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    #for fold in folds:
        #print("Fold {} \n \n".format(fold))
    scores = list()

    predicted = []
    actual = []
    if gradientDescent == GradientDescents.GradientDescent:
        initial_fold = folds[0]
        while(len(folds) > 2):
            fold = folds[1]
            initial_fold.extend(fold)
            folds.remove(fold)
        predicted = algorithm(gradientDescent, pd_ds, dataset, initial_fold, folds[1], *args)
        actual = [row[-1] for row in fold]
    else:
        for fold in folds:
            #print("Test Fold {} \n \n".format(fold))
            train_set = list(folds)
            train_set.remove(fold)
            # print("First Dimensional Length = ",len(train_set))
            # print("Second Dimensional Length = ",len(train_set[0]))
            # print("Third Dimensional Length = ",len(train_set[0][0]))
            train_set = sum(train_set, [])
            # print("Training Set (After Flattening):\nFirst (Flattened) Dimensional Length = {}\nSecond Dimensional Length = {}".format(len(train_set), len(train_set[0])))
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            predicted = algorithm(gradientDescent, pd_ds, dataset, train_set, test_set, *args)
            actual = [row[-1] for row in fold]
            accuracy = accuracy_met(actual, predicted)
            cm = confusion_matrix(actual, predicted)
            print_matrix(cm)
            #confusionmatrix = np.matrix(cm)
            FP = cm.sum(axis=0) - np.diag(cm)
            FN = cm.sum(axis=1) - np.diag(cm)
            TP = np.diag(cm)
            TN = cm.sum() - (FP + FN + TP)
            print('False Positives\n {}'.format(FP))
            print('False Negetives\n {}'.format(FN))
            print('True Positives\n {}'.format(TP))
            print('True Negetives\n {}'.format(TN))
            TPR = TP/(TP+FN)
            print('Sensitivity \n {}'.format(TPR))
            TNR = TN/(TN+FP)
            print('Specificity \n {}'.format(TNR))
            Precision = TP/(TP+FP)
            print('Precision \n {}'.format(Precision))
            Recall = TP/(TP+FN)
            print('Recall \n {}'.format(Recall))
            Acc = (TP+TN)/(TP+TN+FP+FN)
            print('Áccuracy \n{}'.format(Acc))
            Fscore = 2*(Precision*Recall)/(Precision+Recall)
            print('FScore \n{}'.format(Fscore))
            k=cohen_kappa_score(actual, predicted)
            print('Çohen Kappa \n{}'.format(k))
            scores.append(accuracy)
    return scores


# Test the whole algorithm by passing parameters
def test(gradientDescent: GradientDescents = GradientDescents.StochasticGradientDescent):
    # Test Backprop on Seeds dataset
    seed(1)
    # load and prepare data
    filename = 'data-copy.csv' # 'data.csv'
    dataset, pd_ds = load_csv(filename)
    # print_dataset(dataset)

    ds_to_float(dataset)
    # print_dataset(dataset)

    # convert class column to integers
    last_column_index = len(dataset[0]) - 1
    column_to_int(dataset, last_column_index)
    # print_dataset(dataset)

    # normalize input variables
    minmax = min_max(dataset)
    # print(minmax)
    normalize(dataset, minmax)

    # evaluate algorithm
    n_folds = 5
    l_rate = 1e-05 #0.1
    mu=0.001
    n_epoch = 5 #1500
    n_hidden_neurons = 4
    n_hidden_layers = 1
    scores = run_algorithm(gradientDescent, pd_ds, dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden_neurons, n_hidden_layers, mu)

    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

# test(GradientDescents.StochasticGradientDescent)
test(GradientDescents.MiniBatchGradientDescent)
# test(GradientDescents.GradientDescent)
print('end')