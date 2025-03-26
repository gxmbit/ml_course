import numpy as np

def func(x):
    return 0.1 * x**2 - np.sin(x) + 5.

def s(x):
    return np.array([1, x, x**2, x**3]).T

def ax(x, w):
    return s(x) @ w

def dQ(w):
    return (2 / sz) * sum((w.T @ s(x) - func(x)) * s(x) for x in coord_x)

coord_x = np.arange(-5.0, 5.0, 0.1)
coord_y = func(coord_x)

sz = len(coord_x)
eta = np.array([0.1, 0.01, 0.001, 0.0001])
w = np.array([0., 0., 0., 0.])
N = 200

for i in range(N):
    w = w - eta * dQ(w)

Q = 1/sz * sum([(ax(x, w) - func(x))**2 for x in coord_x])

print(w, Q, sep='\n')