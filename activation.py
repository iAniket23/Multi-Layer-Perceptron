# Activation functions for neural networks
import numpy as np
# Activation functions are used to introduce non-linearity to neural networks.
# Without activation functions, neural networks would be linear regression models.
class Activation_ReLU:
    def forward(self, inputs):
        # Forward pass
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

# Softmax activation
# exponential and normalization
class Activation_Softmax:
    def forward(self, inputs):
        # Forward pass
        self.inputs = inputs

        # Now we need to make sure our logic for single sample also works for batch of samples        
        # Get unnormalized probabilities
        # np.exp overflows if we have large numbers
        # So we subtract the maximum value from the inputs
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True))

        # When doing np.sum
        # axis = None -> all values are summed
        # axis = 1 -> row wise, axis = 0 -> column wise
        # keepdims = True -> the output will have the same dimensions as the input
        probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
