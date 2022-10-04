import numpy as np

class Conv:
    def __init__(self, in_channels, out_channels, name, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), initialize_method="random"):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.name = name

        if type(kernel_size) == int:
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if type(stride) == int:
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if type(padding) == int:
            self.padding = (padding, padding)
        else:
            self.padding = padding
        params = self.initialize(initialize_method)
        self.parameters = [params[0], params[1]]
    
    def initialize(self, initialize_method):
        params = [
            np.random.randn(self.kernel_size[0], self.kernel_size[1], self.in_channels, self.out_channels),
            np.zeros((1, 1, 1, self.out_channels))
        ]
        if initialize_method == "random":
            return params
        elif initialize_method == "Xavier":
            params[0] = params[0] * np.sqrt(1 / self.input_size)
            return params
        elif initialize_method == "He":
            params[0] = params[0] * np.sqrt(2 / self.input_size)
            return params
        elif initialize_method == "zero":
            params[0] = np.zeros((self.kernel_size[0], self.kernel_size[1], self.in_channels, self.out_channels))
            return params
        return None

    def target_shape(self, input_shape):
        H = int(1 + (input_shape[0] + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0])
        W = int(1 + (input_shape[1] + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1])
        return H, W

    def pad(self, X, padding, pad_value):
        X_pad = np.pad(X, ((0,0), (padding[0], padding[0]), (padding[1],padding[1]), (0,0)), mode='constant', constant_values = (pad_value,pad_value))
        return X_pad
    
    def single_step_conv(self, a_slice_prev, W, b):
        s = a_slice_prev * W
        Z = np.sum(s) + np.float(b)
        return Z
    
    def forward(self, A_prev):
        W, b = self.parameters[0], self.parameters[1]
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        (fh, fw, n_C_prev, n_C) = W.shape
        strideh, stridew = self.stride[0], self.stride[1]
        n_H, n_W = self.target_shape([n_H_prev, n_W_prev])
        Z = np.zeros((m, n_H, n_W, n_C))
        A_prev_pad = self.pad(A_prev, self.padding, 0)
        for i in range(m):
            a_prev_pad = A_prev_pad[i,:,:,:]
            for h in range(n_H):
                vert_start = h * strideh
                vert_end = vert_start + self.kernel_size[0]
                for w in range(n_W):
                    horiz_start = w * stridew
                    horiz_end = horiz_start + self.kernel_size[1]
                    for c in range(n_C):
                        a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                        weights = W[:,:,:,c]
                        biases = b[:,:,:,c]
                        Z[i, h, w, c] = self.single_step_conv(a_slice_prev, weights, biases)
        return Z, A_prev
    
    def backward(self, dZ, A_prev):
        W, b = self.parameters[0], self.parameters[1]
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        (fh, fw, n_C_prev, n_C) = W.shape
        strideh, stridew = self.stride[0], self.stride[1]
        n_H, n_W = self.target_shape([n_H_prev, n_W_prev])
        dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
        dW = np.zeros((fh, fw, n_C_prev, n_C))
        db = np.zeros((1, 1, 1, n_C))
        A_prev_pad = self.pad(A_prev, self.padding, 0)
        dA_prev_pad = self.pad(dA_prev, self.padding, 0)
        for i in range(m):
            a_prev_pad = A_prev_pad[i, :, :, :]
            da_prev_pad = dA_prev_pad[i, :, :, :]
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):
                        vert_start = h * strideh
                        vert_end = vert_start + fh
                        horiz_start = w * stridew
                        horiz_end = horiz_start + fw
                        a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                        dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                        db[:,:,:,c] += dZ[i, h, w, c]
            dA_prev[i, :, :, :] = da_prev_pad[self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1], :]
        grads = [dW, db]
        return dA_prev, grads
    
    def update(self, optimizer, grads):
        self.parameters = optimizer.update(grads, self.name)
        
    def output_shape(self, X):
        shape_ = X.shape
        shape_[0], shape_[1] = self.target_shape((shape_[0], shape_[1]))
        shape_[2] = self.out_channels
        return shape_