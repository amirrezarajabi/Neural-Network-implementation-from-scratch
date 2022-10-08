import numpy as np

class Sigmoid:
    def __init__(self) -> None:
        pass

    def sig(self, Z):
        A = 1/(1+np.exp(-Z))
        cache = Z
    
        return A

    def sig_backward(self, dA, cache):
        Z = cache
        
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        
        return dZ
    
    def forward(self, Z):
        return self.sig(Z)
    
    def backward(self, dA, cache):
        return self.sig_backward(dA, cache)