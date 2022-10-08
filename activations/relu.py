import numpy as np

class Relu:
    def __init__(self) -> None:
        pass

    def relu_forward(self, Z):
        A = np.maximum(0,Z)
        cache = Z
    
        return A

    def relu_backward(self, dA, cache):
        Z = cache
        
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        
        return dZ
    
    def forward(self, Z):
        return self.relu_forward(Z)
    
    def backward(self, dA, cache):
        return self.relu_backward(dA, cache)