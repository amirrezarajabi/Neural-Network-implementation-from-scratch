import numpy as np

class FC:
    def __init__(self, input_size, output_size, name,initialize_method="random"):
        
        self.input_size = input_size
        self.output_size= output_size
        self.name = name
        params = self.initialize(initialize_method)
        self.parameters = [params[0], params[1]]
        self.input_shape = None
        self.reshaped_input = None

    def initialize(self, initialize_method):
        if initialize_method == "random":
            return [np.random.randn(self.output_size, self.input_size), np.zeros((self.output_size, 1))]
        
        elif initialize_method == "Xavier":
            return [np.random.randn(self.output_size, self.input_size) * np.sqrt(1 / self.input_size), np.zeros((self.output_size, 1))]
        
        elif initialize_method == "He":
            return [np.random.randn(self.output_size, self.input_size) * np.sqrt(2 / self.input_size), np.zeros((self.output_size, 1))]

        elif initialize_method == "zero":
            return [np.zeros((self.output_size, self.input_size)), np.zeros((self.output_size, 1))]
        
        return None
    
    def forward(self, A_prev):
        self.input_shape = A_prev.shape
        A_prev_tmp = np.copy(A_prev)
        if A_prev.ndim == 4:
            BS = A_prev.shape[0]
            A_prev_tmp = A_prev_tmp.reshape(BS, -1).T
        self.reshaped_input = A_prev_tmp.shape

        W, b = self.parameters[0], self.parameters[1]
        Z = W @ A_prev_tmp + b

        return Z
    

    def backward(self, dZ, A_prev):
        A_prev_tmp = np.copy(A_prev)
        if A_prev.ndim == 4:
            BS = A_prev.shape[0]
            A_prev_tmp = A_prev_tmp.reshape(BS, -1).T

        W, b = self.parameters[0], self.parameters[1]
        m = A_prev.shape[1]
        dW = (1 / m) * np.dot(dZ, A_prev_tmp.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        grads = [dW, db]
        if len(self.input_shape) == 4:
            dA_prev = dA_prev.T.reshape(self.input_shape)
        return dA_prev, grads
    
    def update(self, optimizer, grads):
        self.parameters = optimizer.update(grads, self.name)
    
    def output_shape(self, X):
        shape_ = X.shape
        shape_[0] = self.output_size
        return shape_
