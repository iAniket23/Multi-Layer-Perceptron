# importing the library
import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

# initiliaze the seed
nnfs.init()

# np.random.seed(0)
# This is our input -> shape(3,4)
# 3 batches here with each batch having 4 datapoints
# X = [[1,2,3,2.5],
#      [2,5,-1,2],
#      [-1.5,2.7,3.3,-0.8]]


# Layer class for initialization of a layer with weights and biases
class Layer:
    def __init__(self, num_inputs, num_neurons) -> None:
        # the shape of the wight is actually determined by how many inputs are going to be passed in one neuron and number of neurons
        # We also flipped the weights Rows and Column in order to avoid transposing every single time
        # Actually Weights so suppose to be of the shape(num_neurons, num_inputs)
        self.weights = 0.01 * np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))
    
    def forward(self, inputs):
        self.inputs = inputs
        # It outputs datapoint times the weight + bias of that neuron
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

# RELU activation function
class ActivationRELU:
    # one main reason to use RELU is that it is nearly linear
    # we aren't using a exp function becuase it will explode if lots of neurons are there
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

# Softmax
class ActivationSoftmax:
    # takes the unnormalized data and then convert it into normalized data
    def forward(self, inputs):
        self.inputs = inputs
        
        # Unnormalized 
        # Avoiding overflow by subtracting max
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True))
        
        # normalizing it
        probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    
    # Backward pass
    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
        # Calculate sample-wise gradient
        # and add it to the array of sample gradients
        self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

# Loss function
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

# this loss function is better to use when there is a catogorical data and softmax activation for output
class LossCategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        
        # clipping it to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # making a confidence matrix 
        if len(y_true.shape) == 1:
            # when it's not one hot encoded
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            # when it's one hot encoded
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        
        # taking the loss 
        negative_log_liklihoods = -np.log(correct_confidences)

        return negative_log_liklihoods 
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class ActivationSoftmaxLossCategoricalCrossEntropy():
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = ActivationSoftmax()
        self.loss = LossCategoricalCrossEntropy()
    
    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)
 
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Change the Optimizer here
class OptimizerSGD:
    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
    # Update parameters
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases


X, y = spiral_data(samples=100, classes=3)

layer1 = Layer(2, 64)
activation1 = ActivationRELU()

layer2 = Layer(64, 3)
activation2 = ActivationSoftmax()

activationLoss = ActivationSoftmaxLossCategoricalCrossEntropy()

optimizer = OptimizerSGD()

for epoch in range(10001):

    layer1.forward(X)
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)

    loss = activationLoss.forward(layer2.output, y)

    # accuracy
    predictions = np.argmax(activationLoss.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)

    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' + f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f}')

    # Backpropogation
    activationLoss.backward(activationLoss.output, y)
    layer2.backward(activationLoss.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)

    #print gradients
    optimizer.update_params(layer1)
    optimizer.update_params(layer2)
    

