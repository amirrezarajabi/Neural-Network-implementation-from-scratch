import numpy as np

class MaxPool:
    def __init__(self, kernel_size=(1, 1), stride=(1, 1), mode="max"):
        if type(kernel_size) == int:
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if type(stride) == int:
            self.stride = (stride, stride)
        else:
            self.stride = stride

        
        self.mode = mode
    
    def target_shape(self, input_shape):
        H = int(1 + (input_shape[0] - self.kernel_size[0]) / self.stride[0])
        W = int(1 + (input_shape[1] - self.kernel_size[1]) / self.stride[1])
        return H, W

    def forward(self, A_prev):
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        fh, fw = self.kernel_size
        strideh, stridew = self.stride
        n_H, n_W = self.target_shape([n_H_prev, n_W_prev])
        n_C = n_C_prev
        A = np.zeros((m, n_H, n_W, n_C))  
        for i in range(m):
            for h in range(n_H):
                vert_start = h * strideh
                vert_end = vert_start + fh
                for w in range(n_W):
                    horiz_start = w * stridew
                    horiz_end = horiz_start + fw
                    for c in range(n_C):
                        a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end,c]
                        if self.mode == "max":
                            A[i, h, w, c] = np.max(a_prev_slice)
                        elif self.mode == "average":
                            A[i, h, w, c] = np.mean(a_prev_slice)
        return A, A_prev
    
    def create_mask_from_window(self, x):
        mask = x == np.max(x)
        return mask
    
    def distribute_value(self, dz, shape):
    
        (n_H, n_W) = shape
        average = dz / (n_H * n_W)
        a = np.ones(shape) * average
        return a

    def backward(self, dZ, A_prev):
        fh, fw = self.kernel_size
        strideh, stridew = self.stride
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        m, n_H, n_W, n_C = dZ.shape
        dA_prev = np.zeros(A_prev.shape)
        for i in range(m):
            a_prev = A_prev[i]
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):
                        vert_start = h
                        vert_end = vert_start + fh
                        horiz_start = w
                        horiz_end = horiz_start + fw
                        if self.mode == "max":
                            a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                            mask = self.create_mask_from_window(a_prev_slice)
                            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += np.multiply(mask, dZ[i, h, w, c])
                        elif self.mode == "average":
                            dz = dZ[i, h, w, c]
                            shape = (fh, fw)
                            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += self.distribute_value(dz, shape)
        return dA_prev