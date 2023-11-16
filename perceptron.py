# Multi Layer Perceptron

# We will import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# currently we will have 3 neurons 

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
# np.dot() is used to get the dot product of two matrices
# we would have to transpose the inputs matrix in order to get the dot product
layer_outputs = np.dot(weights, np.transpose(inputs)).T + biases

print(layer_outputs)