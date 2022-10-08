from layers.FullyConnected import FC
from layers.Convolution import Conv
from layers.MaxPoolling import MaxPool
from activations.activation import Activation
from optimizers.optimizer import Optimizer
from losses.BinaryCrossEntropy import BCE
from model import Model
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


arch_model = {
    "CONV1": Conv(1, 2, name="CONV1", kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    "RELU1": Activation("relu")(),
    "MAXPOOL1":MaxPool(kernel_size=(2, 2), stride=(2, 2)),
    "LINEAR1": Activation("linear")(),
    "CONV2": Conv(2, 4, name="CONV2", kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    "RELU2": Activation("relu")(),
    "MAXPOOL2":MaxPool(kernel_size=(2, 2), stride=(2, 2)),
    "LINEAR2": Activation("linear")(),
    "FC1": FC(49*4, 16, "FC1"),
    "SIGMOMID1": Activation("sigmoid")(),
    "FC2": FC(16, 1, "FC2"),
    "SIGMOMID2": Activation("sigmoid")(),
}

criterion = BCE()
optimizer = Optimizer("sgd")(arch_model, learning_rate=0.01)

myModel = Model(arch_model, criterion, optimizer)

costs = []
for e in tqdm(range(1, 26)):
    np.random.shuffle(data)
    cost = 0
    for b in range(TOT_STEP):
        x, y = load_batch(b, data, Batch_Size)
        cost += myModel.one_epoch(x, y, Batch_Size) / TOT_STEP
    costs.append(cost)

plt.plot(costs)
plt.show()

X, gt = load_batch(0, data, 4)
Y = myModel.forward(x, 4)[-1]
fig, ax = plt.subplots(2, 2, figsize=(8, 8))
ax[0, 0].imshow(X[0].reshape(28, 28), cmap="gray")
ax[0, 0].set_title(f"{Y[:, 0]} vs {gt[:, 0]}")
ax[0, 1].imshow(X[1].reshape(28, 28), cmap="gray")
ax[0, 1].set_title(f"{Y[:, 1]} vs {gt[:, 1]}")
ax[1, 0].imshow(X[2].reshape(28, 28), cmap="gray")
ax[1, 0].set_title(f"{Y[:, 2]} vs {gt[:, 2]}")
ax[1, 1].imshow(X[3].reshape(28, 28), cmap="gray")
ax[1, 1].set_title(f"{Y[:, 3]} vs {gt[:, 3]}")
plt.show()