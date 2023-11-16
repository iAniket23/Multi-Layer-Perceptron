# Multi Layer Perceptron

# We will import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# getting the batch of inputs, batch size is 3
inputs = [[1, 2, 3, 2.5],
          [2, 5, -1, 2],
          [-1.5, 2.7, 3.3, -0.8]]

# getting the weights, shape is (3,4)
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

# getting the biases
biases = [2, 3, 0.5]

# Now we will calculate the output of the layer
# There are two ways to do this
# layer_outputs = np.dot(weights, np.transpose(inputs)).T + biases
# This requires us to transpose two times to accomodate for the addition of biases

# The other way is to transpose the weights and then multiply it with the inputs
# This is the preferred way
layer_outputs = np.dot(inputs, np.transpose(weights)) + biases

print(layer_outputs)