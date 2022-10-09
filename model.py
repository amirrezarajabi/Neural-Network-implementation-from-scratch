from random import shuffle
import numpy as np
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

## layers
from layers.MaxPoolling import MaxPool
from layers.Convolution import Conv
from layers.FullyConnected import FC

## activations
from activations.relu import Relu
from activations.sigmoid import Sigmoid
from activations.linear import Linear


class Model:
    def __init__(self, model, criterion, optimizer, name=None):
        if name is None:
            self.model = model
            self.criterion = criterion
            self.optimizer = optimizer
            self.layers_name = list(self.model.keys())
        else:
            self.model, self.optimizer, self.criterion, self.layers_name = self.load(name)
    

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
    
    def save(self, name):
        if "saves" not in os.listdir("./"):
            os.mkdir("./saves")
        if name not in os.listdir("./saves/"):
            os.mkdir(f"./saves/{name}")
        DIR = f"./saves/{name}/"
        MODEL = open(f"{DIR}model.rj", "wb")
        OPTIMIZER = open(f"{DIR}optimizer.rj", "wb")
        CRITERION = open(f"{DIR}criterion.rj", "wb")
        LAYERS_NAME = open(f"{DIR}layers_name.rj", "wb")
        pickle.dump(self.model, MODEL)
        pickle.dump(self.optimizer, OPTIMIZER)
        pickle.dump(self.criterion, CRITERION)
        pickle.dump(self.layers_name, LAYERS_NAME)
        MODEL.close()
        OPTIMIZER.close()
        CRITERION.close()
        LAYERS_NAME.close()
    
    def load(self, name):
        DIR = f"./saves/{name}/"
        MODEL = open(f"{DIR}model.rj", "rb")
        OPTIMIZER = open(f"{DIR}optimizer.rj", "rb")
        CRITERION = open(f"{DIR}criterion.rj", "rb")
        LAYERS_NAME = open(f"{DIR}layers_name.rj", "rb")
        model = pickle.load(MODEL)
        optimizer = pickle.load(OPTIMIZER)
        criterion = pickle.load(CRITERION)
        layers_name = pickle.load(LAYERS_NAME)
        MODEL.close()
        OPTIMIZER.close()
        CRITERION.close()
        LAYERS_NAME.close()
        return model, optimizer, criterion, layers_name
    
    def train(self, X, y, X_test = None, y_test=None, Batch_Size=32, epochs=10, shuffling=False, verbose=1, save_after=None):
        costs = []
        val_costs = []
        m = X.shape[0] if X.ndim == 4 else X.shape[1]
        for  e in tqdm(range(1, epochs + 1)):
            order = self.shuffle(m, shuffling)
            cost = 0
            for b in range(m // Batch_Size):
                bx, by = self.load_batch(X, y, Batch_Size, b, order)
                cost += self.one_epoch(X, y, Batch_Size) / (m // Batch_Size)
            costs.append(cost)
            if X_test is not None:
                val_costs.append(self.compute_loss(X_test, y_test, Batch_Size))
            if e % verbose == 0:
                if X_test is not None:
                    print(f"\ntrain cost: {costs[-1]} validation cost: {val_costs[-1]}")
                else:
                    print(f"\ntrain cost: {costs[-1]}")
        if X_test is None:
            plt.plot(list(range(epochs)), costs)
            plt.show()
        if save_after is not None:
            self.save(save_after)
            
    def predict(self, test):
        A0 = test
        AL = self.forward(A0, A0.shape[1])[-1]
        return AL
                

    def compute_loss(self, X, y, Batch_Size):
        m = X.shape[0] if X.ndim == 4 else X.shape[1]
        order = self.shuffle(m, False)
        cost = 0
        for b in range(m // Batch_Size):
            bx, by = self.load_batch(X, y, Batch_Size, b, order)
            tmp = self.forward(bx, Batch_Size)
            AL = tmp[-1]
            cost += self.criterion.compute_cost(AL, y) (m // Batch_Size)
        return cost

    def shuffle(self, m, shuffling):
        order =list(range(m))
        if shuffling is False:
            return order
        np.random.shuffle(order)
        return order
    
    def load_batch(self, X, y, Batch_Size, index, order):
        last_index = min((index + 1) * Batch_Size, len(order))
        batch = order[index * Batch_Size: last_index]
        if X.ndim == 2:
            bx = X[:, batch]
            by = y[:, batch]
            return bx, by
        else:
            bx = X[batch]
            by = y[batch]
            return bx, by