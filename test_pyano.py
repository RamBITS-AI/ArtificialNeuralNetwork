import pyano as dnn
import numpy as np

import utils

from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score

def test_pyano():
	n_folds = 5
	learning_rate = 0.1  # 1e-05
	n_epoch = 1500
	mu = 0.001
	filename = 'data-copy.csv'  # 'data.csv'

	dataset = utils.load_csv(filename)

	utils.ds_to_float(dataset)
	# print_dataset(dataset)

	# convert class column to integers
	last_column_index = len(dataset[0]) - 1
	utils.column_to_int(dataset, last_column_index)
	# print_dataset(dataset)

	# normalize input variables
	minmax = utils.min_max(dataset)
	# print(minmax)
	utils.normalize(dataset, minmax)
	
	folds = utils.cross_validation_split(dataset, n_folds)
	#for fold in folds:
		#print("Fold {} \n \n".format(fold))
	scores = list()

	predicted = []
	actual = []

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
		predicted = train_and_predict(dataset, train_set, test_set, row, learning_rate, n_epoch, mu)
		actual = [row[-1] for row in fold]
		accuracy = utils.accuracy_met(actual, predicted)
		cm = confusion_matrix(actual, predicted)
		utils.print_matrix(cm)
		#confusionmatrix = np.matrix(cm)
		FP = cm.sum(axis=0) - np.diag(cm)
		FN = cm.sum(axis=1) - np.diag(cm)
		TP = np.diag(cm)
		TN = cm.sum() - (FP + FN + TP)
		print('False Positives\n{}'.format(FP))
		print('False Negatives\n{}'.format(FN))
		print('True Positives\n{}'.format(TP))
		print('True Negatives\n{}'.format(TN))
		TPR = TP/(TP+FN)
		print('Sensitivity \n{}'.format(TPR))
		TNR = TN/(TN+FP)
		print('Specificity \n{}'.format(TNR))
		Precision = TP/(TP+FP)
		print('Precision \n{}'.format(Precision))
		Recall = TP/(TP+FN)
		print('Recall \n{}'.format(Recall))
		Acc = (TP+TN)/(TP+TN+FP+FN)
		print('Áccuracy \n{}'.format(Acc))
		Fscore = 2*(Precision*Recall)/(Precision+Recall)
		print('FScore \n{}'.format(Fscore))
		k=cohen_kappa_score(actual, predicted)
		print('Çohen Kappa \n{}'.format(k))
		scores.append(accuracy)

def initialize_network(n_inputs, n_targets):
	n_hidden = 4
	
	input_layer = None
	output_layer = None
	hidden_layer_1 = None
	hidden_layer_2 = None
	hidden_layer_3 = None

	input_layer = dnn.InputLayer(n_inputs, dnn.SigmoidActivation())

	hidden_layer_1 = input_layer.attachToNextLayer(n_inputs, dnn.HiddenLayer(n_hidden, dnn.SigmoidActivation()))
	hidden_layer_2 = hidden_layer_1.attachToNextLayer(n_inputs, dnn.HiddenLayer(n_hidden, dnn.SigmoidActivation()))
	hidden_layer_3 = hidden_layer_2.attachToNextLayer(n_inputs, dnn.HiddenLayer(n_hidden, dnn.SigmoidActivation()))
	# hidden_layer_4 = hidden_layer_3.attachToNextLayer(n_inputs, dnn.HiddenLayer(n_hidden, dnn.SigmoidActivation()))
	# hidden_layer_5 = hidden_layer_4.attachToNextLayer(n_inputs, dnn.HiddenLayer(n_hidden, dnn.SigmoidActivation()))
	# hidden_layer_6 = hidden_layer_5.attachToNextLayer(n_inputs, dnn.HiddenLayer(n_hidden, dnn.SigmoidActivation()))
	# hidden_layer_7 = hidden_layer_6.attachToNextLayer(n_inputs, dnn.HiddenLayer(n_hidden, dnn.SigmoidActivation()))
	# hidden_layer_8 = hidden_layer_7.attachToNextLayer(n_inputs, dnn.HiddenLayer(n_hidden, dnn.SigmoidActivation()))

	output_layer = hidden_layer_3.attachToNextLayer(n_inputs, dnn.OutputLayer(n_targets, dnn.SigmoidActivation()))

	return input_layer, output_layer

# Make a prediction with a network
def predict(input_layer, output_layer, dataset, row):
	outputs = input_layer.full_forward_propagate(row)

	prev_final_output = 0
	final_output = 0
	final_outputs = []

	for neuron in output_layer.neurons:
		final_output = max(prev_final_output, neuron.output)
		prev_final_output = final_output
		final_outputs.append(neuron.output)

	n = final_outputs.index(final_output)

	print("Final Output:", n)

	return n


def train_and_predict(dataset, train, test, row, learning_rate, n_epoch, mu):
	n_input_columns = len(train[0]) - 1 # Number of columns in the training dataset minus the one column which contains the output
	distinct_outputs = set([row[-1] for row in train])
	n_distinct_outputs = len(distinct_outputs)
	input_Layer = None
	output_layer = None
	input_layer, output_layer = initialize_network(n_input_columns, n_distinct_outputs)

	for epoch in range(n_epoch):
		for row in train:
			inputs = row[:-1]

			input_layer.full_forward_propagate(inputs)

			targets = utils.one_hot_encoding(n_distinct_outputs, row[-1])

			# print(targets)

			# output_layer.full_backward_propagate(targets, learning_rate)

			layer = output_layer
			while (layer != None):
				layer.backward_propagate_error(targets)
				layer = layer.prev_layer
			
			layer = input_layer
			while (layer != None):
				layer.update_weights(inputs, learning_rate, mu)
				layer = layer.next_layer

	l = 0
	layer = input_layer
	while (layer.next_layer != None):
		for n, neuron in enumerate(layer.neurons):
			for w, weight in enumerate(neuron.weights):
				print(f"Weight @ Network[{l}][{n}][{w}] = {weight}")
		print(f"Bias @ Network[{l}] = {layer.bias}")
		layer = layer.next_layer
		l += 1

	predictions = list()
	for row in test:
		inputs = row[:-1]
		prediction = predict(input_layer, output_layer, dataset, inputs)
		predictions.append(prediction)

	print('end')
	return predictions

test_pyano()