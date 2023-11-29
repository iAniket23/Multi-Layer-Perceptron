import numpy as np
# calculate the loss of the model
# we calculate it using the cross entropy loss
# Basically we just take log of the softmax output and then take the negative of it

# # Answer is 2
# try_array = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
# print(len(try_array.shape))
# # Answer is 1
# try_array = np.array([1, 2, 3, 4, 5])
# print(len(try_array.shape))
class Loss:
    def calculate(self, output, target):
        # calculate the loss
        sample_losses = self.forward(output, target)

        # calculate the mean loss
        data_loss = np.mean(sample_losses)
        
        return data_loss
    
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        