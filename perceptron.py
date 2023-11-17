# Multi Layer Perceptron

# We will import the necessary libraries and files
import layer
import activation
import nnfs

# See for code: https://gist.github.com/Sentdex/454cb20ec5acf0e76ee8ab8448e6266c
from nnfs.datasets import spiral_data 
nnfs.init()

# getting the data
# 100 points, 3 classes
# each input is a 2 values x coordinate and y coordinate
input_set, y = spiral_data(100, 3) 


# INITIALIZE THE LAYERS AND ACTIVATION FUNCTIONS
# initializing the layer -> Input is 2 because we have 2 features in our input_set
# Layer 1 -> 2 inputs, 3 neurons
hidden_layer_one = layer.Layer_Dense(2, 3)
# initializing the activation function
activation_one = activation.Activation_ReLU()
# Layer 2 -> 3 inputs, 3 neurons
hidden_layer_two = layer.Layer_Dense(3, 3)
# initializing the activation function
activation_two = activation.Activation_Softmax()

# FORWARD PASS
# Forward pass -> Basically it does Wx + b
hidden_layer_one.forward(input_set)
# activation function
activation_one.forward(hidden_layer_one.output)

# Layer 2
hidden_layer_two.forward(activation_one.output)
activation_two.forward(hidden_layer_two.output)


print(activation_two.output)