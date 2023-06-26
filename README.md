# Contributions
    Snehanshu Saha Sir from BITS Pilani
        For Teaching the mathematical concepts and also showing with examples on how to develop a Neural Network.
        For guiding in many ways and sharing his wisdom in the art of machine learning.
        And for sharing a working Artificial MultiLayerPerceptron implementation in Python.
    Ramachandran C - M.Tech Student of Snehanshu Saha Sir in BITS Pilani and Primary Author of Pyano.

# Pyano - A Deep Neural Network Framework
A deep artificial neural network framework that utilizes back propagation for training. It has a total of 5  layers of neurons. The hidden layers have 4 each, the input layer has 3 and the output layer has 6 neurons (as per the number of distinct classes in the given dataset).

# Deep Neural Network
The Layers in the Neural Network consist of:  
    1. Single Input Layer with 3 neurons.  
    2. Three Hidden Layers with 4 neurons each.  
    3. Single Output Layer with 6 neurons.

It represents a 6 class classification problem.

Weights are initialized randomly.

The Activation function is a sigmoid.

![](sigmoid.png)

The Dataset has samples with 3 features each.

![](NeuralNetwork.png)

# Objective
Define and implement an Object-Oriented Deep Neural Network Framework (capable of implementing DNNs with multiple hidden layers) that predicts classes for a test set with decently accurate results.

Implement Back propagation algorithm to update weights.

Compute the forward and backward passes with error gradients for every input sample.

Repeat the process for every sample in the training set.

# Code Overview
Pyano is a framework for building and training deep neural networks. Here's a high-level overview of the main components:

1. **Pyano.py**: This file contains the core classes and methods for building a neural network. It includes classes for different types of layers (InputLayer, HiddenLayer, OutputLayer) and neurons (ActivationNeuron, WeightedNeuron), as well as different activation functions (SigmoidActivation, SoftMaxActivation, TanhActivation, SBAFActivation). It also includes methods for forward and backward propagation.

2. **Play_pyano.py**: This file contains functions for initializing a network, training it, and making predictions. It uses the classes and methods defined in Pyano.py.

3. **Utils.py**: This file contains utility functions that are used throughout the project. These include functions for loading and normalizing data, generating weights and biases, and calculating activation function derivatives.

4. **Core/layers.py**: This file contains classes for building a network (Network, Sequential). These classes use the layer and neuron classes defined in Pyano.py to build a full neural network.

5. **Data.csv and data-copy.csv**: These files likely contain the data that is used to train and test the neural network.

6. **Pyano_stash.py**: This file appears to be a backup or older version of Pyano.py, as it contains similar classes and methods.

## Pyano.py
The `pyano.py` file is the core of the Pyano Deep Neural Network Framework. It contains several classes and methods that are used to construct and train a neural network. Here's a detailed explanation of each:

1. **Classes**:
   - `Activation`: This is likely an abstract base class (ABC) for different types of activation functions.
   - `ActivationLayer`, `WeightedLayer`: These classes represent layers in the neural network that apply an activation function and weighted sum respectively.
   - `ActivationNeuron`, `WeightedNeuron`: These classes represent neurons in the neural network that apply an activation function and weighted sum respectively.
   - `GradientDescents`: This class likely represents different types of gradient descent algorithms for training the network.
   - `HiddenLayer`, `InputLayer`, `OutputLayer`: These classes represent different types of layers in the neural network.
   - `NeuralLayer`, `Neuron`: These are likely abstract base classes for layers and neurons in the neural network.
   - `SBAFActivation`, `SigmoidActivation`, `SoftMaxActivation`, `TanhActivation`: These classes represent different types of activation functions.

2. **Methods**:
   - `__init__`: This is the constructor method for a class. It's called when an object of the class is created.
   - `activate`: This method is used to apply the activation function to the inputs of a neuron.
   - `attachToNextLayer`: This method is used to connect one layer of the network to the next.
   - `backward_propagate_error`: This method is used to propagate the error from the output layer back to the input layer during training.
   - `forward_propagate`: This method is used to propagate the inputs from the input layer to the output layer.
   - `full_forward_propagate`: This method is used to propagate the inputs from the input layer to the output layer for all layers in the network.
   - `get_activation`, `get_derivative`: These methods are used to get the activation function and its derivative.
   - `update_weights`: This method is used to update the weights of the neurons based on the error.

3. **Variables**:
   - The variables are used to store various pieces of information such as the inputs to the network (`inputs`), the weights of the neurons (`weights`), the bias of the neurons (`bias`), the learning rate (`learning_rate`), and the outputs of the network (`output`).
  
