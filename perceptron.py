# Multi Layer Perceptron

# We will import the necessary libraries and files
import layer

# getting the batch of inputs, batch size is 3
input_set =  [[1, 2, 3, 2.5],
             [2, 5, -1, 2],
             [-1.5, 2.7, 3.3, -0.8]]

# initializing the layer 
hidden_layer_one = layer.Layer_Dense(4, 5)
hidden_layer_two = layer.Layer_Dense(5, 2)

# Forward pass
hidden_layer_one.forward(input_set)
hidden_layer_two.forward(hidden_layer_one.output)

print(hidden_layer_two.output)