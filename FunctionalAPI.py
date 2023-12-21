"""
A usual way to make neaural networks, we will compare our result with this file

Dataset: MNIST

"""
import tensorflow as tf
import keras
from keras import layers, Model
import numpy as np

# Define the input layer
# 784 is the number of features which are the number pixels, left the batch size as None
model_input = keras.Input(shape=(784,)) 

layer_one_output = layers.Dense(64, activation='relu')(model_input)

# Layer two with 32 neurons and relu activation
layer_two_output = layers.Dense(64, activation='relu')(layer_one_output)

# Layer three with 10 neurons(category) and softmax activation
# mnist has 10 categories
model_output = layers.Dense(10, activation='softmax')(layer_two_output)

# Create a model
model = Model(inputs=model_input, outputs=model_output, name='mnist_model_functional')

# model.summary()
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data (these are NumPy arrays)
# divide by 255 to normalize the data between 0 and 1, why 255? because it is the max value of a pixel
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

model.compile(
    # from logits means that the output of the model is not normalized
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.RMSprop(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)

# Evaluate the model
test_scores = model.evaluate(x_test, y_test)


print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])