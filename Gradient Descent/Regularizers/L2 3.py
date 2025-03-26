import numpy as np


def func(x):
    return 0.1 * x + 0.1 * x ** 2 - 0.5 * np.sin(2*x) + 1 * np.cos(4*x) + 10


x = np.arange(-3.0, 4.1, 0.1)
y = np.array(func(x))

N = 22
lm = 20

X = np.array([[a ** n for n in range(N)] for a in x])
IL = np.array([[lm if i == j else 0 for j in range(N)] for i in range(N)])
IL[0][0] = 0

X_train = X[::2]
Y_train = y[::2]

w = np.linalg.inv(X_train.T @ X_train + IL) @ X_train.T @ Y_train

a = X @ w

Q = np.mean((a - y) ** 2)

print(w, Q)