from layers.FullyConnected import FC
from activations.activation import Activation
from optimizers.optimizer import Optimizer
from losses.BinaryCrossEntropy import BCE
import matplotlib.pyplot as plt
from model import Model
import numpy as np
from tqdm import tqdm

def load_planar_dataset():
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T

    return X, Y

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)

X, y = load_planar_dataset()
plt.scatter(X[0, :], X[1, :], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()


arch_model = {
    "FC1":FC(2, 5, "FC1", "He"),
    "S1": Activation("sigmoid")(),
    "FC2":FC(5, 1, "FC2", "He"),
    "S2": Activation("sigmoid")()
}

optimizer = Optimizer("sgd")(arch_model, learning_rate=0.01)

criterion = BCE()

model = Model(arch_model, criterion, optimizer)

Batch_Size = 40

costs = []
for e in tqdm(range(1, 10001)):
    cost = 0
    for b in range(X.shape[1] // Batch_Size):
        A0 = X[:, b * Batch_Size:(b + 1) * Batch_Size]
        by = y[:, b * Batch_Size:(b + 1) * Batch_Size]
        cost += model.one_epoch(A0, by, Batch_Size) / (X.shape[1] // Batch_Size)
    costs.append(cost)

plt.plot(costs)
plt.show()

def predict(test):
    A0 = test
    A2 = model.forward(A0, A0.shape[1])[-1]
    predictions = np.round(A2)
    return predictions

plot_decision_boundary(lambda x: predict(x.T), X, y)
plt.show()