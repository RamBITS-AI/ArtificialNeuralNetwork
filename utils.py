""" 
    utils.py
    
	Created on 22-March-2021 @ 11:15 AM IST

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
from random import seed
from random import randrange
from random import random
from math import exp

import csv
import numpy as np

# Load a CSV file
def load_csv(filename):
    lines = csv.reader(open(filename, 'r'))
    dataset = list(lines)
    return dataset


# Convert string column to float
def column_to_float(dataset, column):
    for row in dataset:
        try:
            row[column] = float(row[column])
        except ValueError:
            print("Error with row",column,":",row[column])
            pass


def ds_to_float(dataset):
    col_count = len(dataset[0])
    for col in range(col_count):
        column_to_float(dataset, col)


# Rescale dataset columns to the range 0-1
def normalize(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-1):
            min = minmax[i]['min']
            max = minmax[i]['max']
            row[i] = (row[i] - min) / (max - min)


# Find the min and max values for each column
def min_max(dataset):
    zipped_ds = zip(*dataset)
    minmax = [{'min': min(column), 'max': max(column)} for column in zipped_ds]
    return minmax


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


# One Hot Coding
def one_hot_encoding(n_possible_outcomes, outcome):
    one_hot = [0 for i in range(n_possible_outcomes)]
    one_hot[outcome] = 1
    return one_hot


# Print a Formatted Matrix
def print_matrix(matrix, padding = 4):
    print('\n'.join([''.join([f'{item:{padding}}' for item in row]) for row in matrix]))
 
 
# Print a Formatted Matrix
def print_network(network, padding = 4):
    pass
 
def print_outputs_of_neurons(network, padding = 4):
    pass

# One Hot Encoding
def one_hot_encoding(n_possible_outcomes, outcome):
    one_hot = [0 for i in range(n_possible_outcomes)]
    one_hot[outcome] = 1
    return one_hot

def generate_weights(count):
	return [random() for i in range(count)]

def generate_zeros(count):
	return [0 for i in range(count)]

def generate_bias():
	return random()

def softmax(activation, weights):
    exp_inputs = [exp(weight) for weight in weights]
    sm = (exp(activation) - max(exp_inputs)) / sum(exp_inputs)
    return sm
    
    # eZ = np.exp(weights)
    # sm = eZ / np.sum(eZ)
    # return sm

    # exps = np.exp(weights)
    # sums = np.sum(exps)
    # return np.divide(exps, sums)

    # exp_inputs = [exp(input) for input in inputs]
    # return (exp(activation) - max(exp_inputs)) / sum(exp_inputs)

def softmax_derivative(input, output, outputs):
    S = np.array(outputs, dtype=float)
    S_vector = S.reshape(S.shape[0],1)
    S_matrix = np.tile(S_vector,S.shape[0])
    derivative = np.diag(S) - (S_matrix * np.transpose(S_matrix))
    derivative_value = derivative[outputs.index(output)]
    return derivative_value
    # return output * (1 - output)

def tanh(activation, weights):
	return (exp(activation) - exp(-activation)) / (exp(activation) + exp(-activation))

def tanh_derivative(input, output, outputs):
    return 1 - pow(output, 2)

def sigmoid(activation, weights):
    return 1.0 / (1.0 + exp(-activation))

def sigmoid_derivative(input, output, outputs):
    return output * (1 - output)

def sbaf(activation, weights):
    k = 0.91
    alpha = 0.5
    return (1/(1 + k * pow(activation, alpha) * pow((1 - activation), (1 - alpha))))

def sbaf_derivative(input, output, outputs):
    alpha = 0.5
    return (((output - 1) * output) / (input * (1 - input))) * (alpha - input)