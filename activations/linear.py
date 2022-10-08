import numpy as np

class Linear:
    def __init__(self) -> None:
        pass

    def linear_forward(self, Z):
        A = Z
        cache = Z
    
        return A

    def linear_backward(self, dA, cache):
        Z = cache
        
        dZ = dA
        
        return dZ
    
    def forward(self, Z):
        return self.linear_forward(Z)
    
    def backward(self, dA, cache):
        return self.linear_backward(dA, cache)