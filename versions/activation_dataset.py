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

# getting the batch of inputs, batch size is 3
# input_set =  [[1, 2, 3, 2.5],
#              [2, 5, -1, 2],
#              [-1.5, 2.7, 3.3, -0.8]]

# initializing the layer -> Input is 2 because we have 2 features in our input_set
hidden_layer_one = layer.Layer_Dense(2, 5)
# hidden_layer_two = layer.Layer_Dense(5, 2)

# initializing the activation function
activation_one = activation.Activation_ReLU()

# Forward pass -> Basically it does Wx + b
hidden_layer_one.forward(input_set)

# activation function
activation_one.forward(hidden_layer_one.output)


# hidden_layer_two.forward(hidden_layer_one.output)

print(activation_one.output)