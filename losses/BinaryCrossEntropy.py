import numpy as np

class BCE:
    def __init__(self):
        pass

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        # Y.shape: (n, batch_size)
        cost = (-1/m) * (np.dot(Y, np.log(AL).T) + np.dot((1-Y), np.log(1-AL).T))
        return np.squeeze(cost)
    
    def backward(self, AL, Y):
        # Y.shape: (n, batch_size)
        return - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))