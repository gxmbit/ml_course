import numpy as np


def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 3


coord_x = np.arange(-4.0, 6.0, 0.1)
coord_y = func(coord_x)
K = 10
X = np.array([[xx**i for i in range(K)] for xx in coord_x])
Y = coord_y

X_train = X[::2]
Y_train = Y[::2]

F = 1/len(X_train) * X_train.T @ X_train
L, W = np.linalg.eig(F)
WW = sorted(zip(L, W), key=lambda lx: lx[0], reverse=False)
WW = np.array([w[1] for w in WW])

G = X @ WW.T
G = G[:, :7]

XX_train = G[::2]

w = np.linalg.inv(XX_train.T @ XX_train) @ XX_train.T @ Y_train

predict = G @ w
print(predict)
