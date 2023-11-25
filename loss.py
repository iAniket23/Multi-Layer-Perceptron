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
    def __init__(self, output, target):
        self.output = output
        self.target = target
        self.loss = self.cross_entropy_loss()
        