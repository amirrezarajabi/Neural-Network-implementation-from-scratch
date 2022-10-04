from layers.FullyConnected import FC
from layers.Convolution import Conv
from layers.MaxPoolling import MaxPool
from activations.activation import Activation
from optimizers.optimizer import Optimizer
from losses.BinaryCrossEntropy import BCE
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
import os
DIR = "./datasets/MNIST/"
DIR0 = DIR + "0/"
DIR1 = DIR + "1/"
FILES0 = [DIR0 + f for f in os.listdir(DIR0)]
FILES1 = [DIR1 + f for f in os.listdir(DIR1)]

def load_data(path, label):
    return  [np.expand_dims(np.array(Image.open(path)) / 255., axis=-1), label]

data = []
for f in FILES0:
    data.append(load_data(f, 0))
for f in FILES1:
    data.append(load_data(f, 1))

Batch_Size = 47
N = len(data)
TOT_STEP = N // Batch_Size

def load_batch(b, data, BS):
    tmp = data[b * BS:(b + 1) * BS]
    X, y = [], []
    for xy in tmp:
        X.append(xy[0])
        y.append(xy[1])
    return np.array(X), np.array(y).reshape(1, -1)

model = {
    "CONV1": Conv(1, 2, name="CONV1", kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    "MAXPOOL1":MaxPool(kernel_size=(2, 2), stride=(2, 2)),
    "CONV2": Conv(2, 4, name="CONV2", kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    "MAXPOOL2":MaxPool(kernel_size=(2, 2), stride=(2, 2)),
    "FC1": FC(49*4, 16, "FC1"),
    "FC2": FC(16, 1, "FC2"),
}

SIGMOID = Activation("sigmoid")()
RELU = Activation("relu")()

criterion = BCE()
optimizer = Optimizer("sgd")(model, learning_rate=0.01)

costs = []
for e in tqdm(range(1, 6)):
    np.random.shuffle(data)
    cost = 0
    for b in range(TOT_STEP):
        A0, y = load_batch(b, data, Batch_Size)
        Z1, A0 = model["CONV1"].forward(A0)
        A1, Z1 = RELU.forward(Z1)
        Z2, A1 = model["MAXPOOL1"].forward(A1)
        A2 = Z2
        Z3, A2 = model["CONV2"].forward(A2)
        A3, Z3 = RELU.forward(Z3)
        Z4, A3 = model["MAXPOOL2"].forward(A3)
        A4 = Z4
        A4 = A4.reshape((Batch_Size, -1)).T
        Z5, A4 = model["FC1"].forward(A4)
        A5, Z5 = SIGMOID.forward(Z5)
        Z6, A5 = model["FC2"].forward(A5)
        A6, Z6 = SIGMOID.forward(Z6)
        loss = criterion.compute_cost(A6, y)
        cost += loss / TOT_STEP
        dA6 = criterion.backward(A6, y)
        dZ6 = SIGMOID.backward(dA6, Z6)
        dA5, grads6 = model["FC2"].backward(dZ6, A5)
        dZ5 = SIGMOID.backward(dA5, Z5)
        dA4, grads5 = model["FC1"].backward(dZ5, A4)
        dA4 = dA4.reshape((Batch_Size, 7, 7, 4))
        dZ4 = dA4
        dA3 = model["MAXPOOL2"].backward(dZ4, A3)
        dZ3 = RELU.backward(dA3, Z3)
        dA2, grads3 = model["CONV2"].backward(dZ3, A2)
        dZ2 = dA2
        dA1 = model["MAXPOOL1"].backward(dZ2, A1)
        dZ1 = RELU.backward(dA1, Z1)
        dA0, grads1 = model["CONV1"].backward(dZ1, A0)
        model["FC2"].update(optimizer, grads6)
        model["FC1"].update(optimizer, grads5)
        model["CONV2"].update(optimizer, grads3)
        model["CONV1"].update(optimizer, grads1)
    costs.append(cost)

plt.plot(costs)
plt.show()

X, gt = load_batch(0, data, 4)

Z1, A0 = model["CONV1"].forward(X)
A1, Z1 = RELU.forward(Z1)
Z2, A1 = model["MAXPOOL1"].forward(A1)
A2 = Z2
Z3, A2 = model["CONV2"].forward(A2)
A3, Z3 = RELU.forward(Z3)
Z4, A3 = model["MAXPOOL2"].forward(A3)
A4 = Z4
A4 = A4.reshape((4, -1)).T
Z5, A4 = model["FC1"].forward(A4)
A5, Z5 = SIGMOID.forward(Z5)
Z6, A5 = model["FC2"].forward(A5)
Y, Z6 = SIGMOID.forward(Z6)

fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(X[0].reshape(28, 28), cmap="gray")
ax[0, 0].set_title(f"{Y[:, 0]} vs {gt[:, 0]}")
ax[0, 1].imshow(X[1].reshape(28, 28), cmap="gray")
ax[0, 1].set_title(f"{Y[:, 1]} vs {gt[:, 1]}")
ax[1, 0].imshow(X[2].reshape(28, 28), cmap="gray")
ax[1, 0].set_title(f"{Y[:, 2]} vs {gt[:, 2]}")
ax[1, 1].imshow(X[3].reshape(28, 28), cmap="gray")
ax[1, 1].set_title(f"{Y[:, 3]} vs {gt[:, 3]}")
plt.show()
