# importing the library
import numpy as np
np.random.seed(0)
# This is our input -> shape(3,4)
# 3 batches here with each batch having 4 datapoints
X = [[1,2,3,2.5],
      [2,5,-1,2],
      [-1.5,2.7,3.3,-0.8]]

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

number_inputs_layer1 = 4
number_neurons_layer1 = 5
layer1 = Layer(number_inputs_layer1, number_neurons_layer1)

# this input should be equal to the output of layer 1
number_inputs_layer2 = 5
number_neurons_layer2 = 2
layer2 = Layer(number_inputs_layer2, number_neurons_layer2)

layer1.forward(X)
print("Layer 1 \n",layer1.output)

layer2.forward(layer1.output)
print("Layer 2 \n",layer2.output)