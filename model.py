import numpy as np

## layers
from layers.MaxPoolling import MaxPool
from layers.Convolution import Conv
from layers.FullyConnected import FC

## activations
from activations.relu import Relu
from activations.sigmoid import Sigmoid
from activations.linear import Linear

class Model:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.layers_name = list(self.model.keys())
    

    def isLayer(self, layer):
        return type(layer) == MaxPool or type(layer) == Conv or type(layer) == FC
    
    def isActivation(self, layer):
        return type(layer) == Sigmoid or type(layer) == Linear or type(layer) == Relu
    
    def isMaxpool(Self, layer):
        return type(layer) == MaxPool

    def forward(self, x, Batch_Size):
        tmp = []
        A = x
        for i in range(0, len(self.layers_name), 2):
            Z = self.model[self.layers_name[i]].forward(A)
            tmp.append(np.copy(Z))
            A = self.model[self.layers_name[i + 1]].forward(Z)
            tmp.append(np.copy(A))
        return tmp
    
    def backward(self, dAL, tmp, x):
        dA = dAL
        grads = {}
        for i in range(len(tmp) - 1, -1, -2):
            if i > 2:
                Z,A = tmp[i - 1], tmp[i - 2]
            else:
                Z, A = tmp[i - 1], x
            dZ = self.model[self.layers_name[i]].backward(dA, Z)
            dA, grad = self.model[self.layers_name[i - 1]].backward(dZ, A)
            grads[self.layers_name[i - 1]] = grad

        return grads
    
    def update(self, grads):
        for name in self.layers_name:
            if self.isLayer(self.model[name]) and not self.isMaxpool(self.model[name]):
                self.model[name].update(self.optimizer, grads[name])
    

    def one_epoch(self, x, y, Batch_Size):
        tmp = self.forward(x, Batch_Size)
        AL = tmp[-1]
        loss = self.criterion.compute_cost(AL, y)
        dAL = self.criterion.backward(AL, y)
        grads = self.backward(dAL, tmp, x)
        self.update(grads)
        return loss