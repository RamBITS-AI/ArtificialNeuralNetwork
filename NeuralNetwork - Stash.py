# Backprop on the Vowel Dataset
from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import numpy as np
import csv
 
# Load a CSV file
def load_csv(filename):
    lines = csv.reader(open(filename, 'r'))
    dataset = list(lines)
    return dataset


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
    print('\n'.join(['\n'.join([f'Weights => {item["weights"]} \n Prev => {item["prev"]}' for item in row]) for row in network]))
    print("\n\n\n")
 
# Initialize a network
def initialize_network(n_input_columns, n_hidden, n_distinct_possible_outputs):
    network = list()
    input_layer = [{'weights':[random() for i in range(n_input_columns + 1)], 'prev':[0 for i in range(n_input_columns + 1)]} for i in range(n_hidden)]
    network.append(input_layer)
    print("Single Layer:")
    print_network(network)
    hidden_layer = [{'weights':[random() for i in range(n_input_columns + 1)], 'prev':[0 for i in range(n_input_columns + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    print("Two Layers:")
    print_network(network)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)], 'prev':[0 for i in range(n_hidden + 1)]} for i in range(n_distinct_possible_outputs)]
    network.append(output_layer)
    print("Three Layers:")
    print_network(network)
    return network
 
# One Hot Coding
def one_hot_coding(n_possible_outcomes, outcome):
    one_hot = [0 for i in range(n_possible_outcomes)]
    one_hot[outcome] = 1
    return one_hot

 # Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs, mu):
    for epoch in range(n_epoch):
        for row in train:
            outputs = forward_propagate(network, row)
            #print(network)
            expected = one_hot_coding(n_outputs, row[-1])
            #print("expected row{}\n".format(expected))
            backward_propagate_error(network, expected)
            # print_delta_and_output(network)
            update_weights(network, row, l_rate, mu)
            # print_delta_and_output(network)


# Calculate neuron activation for an input
def activate(weights, inputs):
    bias = weights[-1]
    activation = 0
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    activation += bias
    return activation
 

# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))
 

# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs
 

# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)
 

def print_delta_and_output(network):
    l = 0
    for layer in network:
        n = 0
        for neuron in layer:
            if 'delta' in neuron.keys():
                print(f"network[{l}][{n}]['delta'] = {neuron['delta']}")
            if 'output' in neuron.keys():
                print(f"network[{l}][{n}]['output'] = {neuron['output']}")
            n += 1
        l += 1


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))): # From Output Layer to Input Layer
        layer = network[i]
        errors = list()
        if i != len(network)-1: # If it's not the Output Layer
            # Hidden Layers and Input Layer
            for j in range(len(layer)):
                error = 0.0
                for next_neuron in network[i + 1]:
                    error += (next_neuron['weights'][j] * next_neuron['delta'])
                errors.append(error)
        else:
            # Output Layer
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        # For all Layers
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
 

# Update network weights with error
def update_weights(network, row, l_rate, mu):
    for i in range(len(network)):
        inputs = row[:-1]        
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                temp = l_rate * neuron['delta'] * inputs[j] + mu * neuron['prev'][j]
                
                neuron['weights'][j] += temp
                #print("neuron weight{} \n".format(neuron['weights'][j]))
                neuron['prev'][j] = temp
            temp = l_rate * neuron['delta'] + mu * neuron['prev'][-1]
            neuron['weights'][-1] += temp
            neuron['prev'][-1] = temp
                
 
# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))
 

# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden, mu):
    n_input_columns = len(train[0]) - 1 # Number of columns in the training dataset minus the one column which contains the output
    distinct_outputs = set([row[-1] for row in train])
    n_distinct_outputs = len(distinct_outputs)
    network = initialize_network(n_input_columns, n_hidden, n_distinct_outputs)
    train_network(network, train, l_rate, n_epoch, n_distinct_outputs, mu)
    #print("network {}\n".format(network))
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return(predictions)


# Evaluate an algorithm using a cross validation split
def run_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    #for fold in folds:
        #print("Fold {} \n \n".format(fold))
    scores = list()
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
        predicted = algorithm(train_set, test_set, *args)
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
def test():
    # Test Backprop on Seeds dataset
    seed(1)
    # load and prepare data
    filename = 'data.csv'
    dataset = load_csv(filename)
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
    l_rate = 0.1
    mu=0.001
    n_epoch = 3 # 1500
    n_hidden = 4
    scores = run_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden, mu)

    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

test()