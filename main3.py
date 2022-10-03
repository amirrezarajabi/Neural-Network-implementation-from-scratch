from layers.FullyConnected import FC
from activations.activation import Activation
from optimizers.optimizer import Optimizer
from losses.MeanSquaredError import MSE
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

X = np.random.rand(3, 128)
y1 = 3 * X[0] - 2 * X[1] + X[2]
y2 = X[0] ** 2 + 4 * X[1] - X[2] ** 2
y = np.vstack([y1, y2])

model = {
    "FC1":FC(3, 4, "FC1", "He"),
    "FC2":FC(4, 2, "FC2", "He")
}

optimizer = Optimizer("sgd")(model)

criterion = MSE()

SIGMOID = Activation("sigmoid")()

Batch_Size = 32

costs = []
for e in tqdm(range(1, 100001)):
    cost = 0
    for b in range(X.shape[1] // Batch_Size):
        A0 = X[:, b * Batch_Size:(b + 1) * Batch_Size]
        by = y[:, b * Batch_Size:(b + 1) * Batch_Size]
        Z1, A0 = model["FC1"].forward(A0)
        A1, Z1 = SIGMOID.forward(Z1)
        Z2, A1 = model["FC2"].forward(A1)
        A2 = Z2
        loss = criterion.compute_cost(A2, by)
        dA2 = criterion.backward(A2, by)
        dZ2 = dA2
        dA1, grads2 = model["FC2"].backward(dZ2, A1)
        dZ1 = SIGMOID.backward(dA1, Z1)
        dA0, grads1 = model["FC1"].backward(dZ1, A0)
        model["FC2"].update(optimizer, grads2)
        model["FC2"].update(optimizer, grads2)
        cost += loss / (X.shape[1] // Batch_Size)
    costs.append(cost)

plt.plot(costs)
plt.show()

x_t = np.random.rand(3, 1)
y_t, _ = model["FC1"].forward(x_t)
y_t, _ = SIGMOID.forward(y_t)
y_t, _ = model["FC2"].forward(y_t)

print(x_t)
print("#" * 15)
print(y_t)
print("VS")
gt1 = 3 * x_t[0] - 2 * x_t[1] + x_t[2]
gt2 = x_t[0] ** 2 + 4 * x_t[1] - x_t[2] ** 2
gt = np.vstack([gt1, gt2])
print(gt)