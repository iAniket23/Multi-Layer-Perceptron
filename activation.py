# Activation functions for neural networks

import numpy as np
# Activation functions are used to introduce non-linearity to neural networks.
# Without activation functions, neural networks would be linear regression models.
class Activation_ReLU:
    def forward(self, inputs):
        # Forward pass
        self.inputs = inputs
        self.output = np.maximum(0, inputs)