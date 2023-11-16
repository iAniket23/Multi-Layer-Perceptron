# Multi Layer Perceptron

# We will import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# currently we will have 3 neurons 

# getting the inputs
inputs = [1, 2, 3, 2.5]

# getting the weights
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

# getting the biases
biases = [2, 3, 0.5]

# Output of current layer
layer_outputs = np.dot(weights, inputs) + biases

print(layer_outputs)