# importing the library
import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data  # See for code: https://gist.github.com/Sentdex/454cb20ec5acf0e76ee8ab8448e6266c

nnfs.init()

# np.random.seed(0)
# This is our input -> shape(3,4)
# 3 batches here with each batch having 4 datapoints
# X = [[1,2,3,2.5],
#       [2,5,-1,2],
#       [-1.5,2.7,3.3,-0.8]]


# Layer class for initialization of a layer with weights and biases
class Layer:
    def __init__(self, num_inputs, num_neurons) -> None:
        # the shape of the wight is actually determined by how many inputs are going to be passed in one neuron and number of neurons
        # We also flipped the weights Rows and Column in order to avoid transposing every single time
        # Actually Weights so suppose to be of the shape(num_neurons, num_inputs)
        self.weights = 0.10 * np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# RELU
class ActivationRELU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# Softmax
class ActivationSoftmax:
    def forward(self, inputs):
        # Avoiding overflow by subtracting max
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True))
        probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

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