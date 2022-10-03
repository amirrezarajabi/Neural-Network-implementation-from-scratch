class SGD:
    def __init__(self, layers_list, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.layers = layers_list
    
    def update(self, grads, name, epoch=1):
        layer = self.layers[name]
        params = []
        for index in range(len(grads)):
            params.append(layer.parameters[index] - self.learning_rate * grads[index])
        return params


