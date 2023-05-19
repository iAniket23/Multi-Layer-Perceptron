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
        self.weights = 0.10 * np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))
    
    def forward(self, inputs):
        # It outputs datapoint times the weight + bias of that neuron
        self.output = np.dot(inputs, self.weights) + self.biases

# RELU activation function
class ActivationRELU:
    # one main reason to use RELU is that it is nearly linear
    # we aren't using a exp function becuase it will explode if lots of neurons are there
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# Softmax
class ActivationSoftmax:
    # takes the unnormalized data and then convert it into normalized data
    def forward(self, inputs):
        # Unnormalized 
        # Avoiding overflow by subtracting max
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True))
        # normalizing it
        probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# Loss function
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

# this loss function is better to use when there is a catogorical data and softmax activation for output
class Loss_CategoricalCrossEntropy(Loss):
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

X, y = spiral_data(samples=100, classes=3)

layer1 = Layer(2, 3)
activation1 = ActivationRELU()

layer2 = Layer(3, 3)
activation2 = ActivationSoftmax()

layer1.forward(X)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

print(activation2.output[:5])

loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, y)

print("Loss:", loss)