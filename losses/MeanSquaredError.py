import numpy as np

class MSE:
    def __init__(self):
        pass

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        # Y.shape: (n, batch_size)
        cost = np.mean(np.square(Y - AL)) / 2
        return np.squeeze(cost)
    
    def backward(self, AL, Y):
        # Y.shape: (n, batch_size)
        return AL - Y