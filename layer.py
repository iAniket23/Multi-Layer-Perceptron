# Description: This file contains the Layer class which is used to create layers for our neural network.
import numpy as np

# for weight initialization
# np.random.seed(0)

# Layers for our neural network
# Each layer is initialized with weights and biases given the number of inputs and neurons
# Forward pass is done by multiplying the inputs with the weights and adding the biases
class Layer_Dense:
    def __init__(self, number_of_inputs, number_of_neurons):
        # Initialize weights and biases

        # we initialized weights in a way to avoid transposing it later during forward pass
        self.weights = 0.10 * np.random.randn(number_of_inputs, number_of_neurons)
        self.biases = np.zeros((1, number_of_neurons))
    
    def forward(self, inputs):
        # Forward pass
        self.inputs = inputs

        # weights already transposed, shape of weight is (num_inputs, num_neurons) instead of (num_neurons, num_inputs)
        self.output = np.dot(inputs, self.weights) + self.biases