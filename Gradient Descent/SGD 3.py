import numpy as np

def func(x):
    return 0.5 * x**2 - 0.1 * 1/np.exp(-x) + 0.5 * np.cos(2*x) - 2.

def loss(w, x, y):
    return (w @ x - y) ** 2

def df(w, x, y):
    return 2 * (w @ x - y) * x

coord_x = np.arange(-5.0, 5.0, 0.1)
coord_y = func(coord_x)

sz = len(coord_x)
eta = np.array([0.01, 0.001, 0.0001, 0.01, 0.01])
w = np.array([0., 0., 0., 0., 0.])
N = 500
lm = 0.02

x_train = np.array([[1, x, x**2, np.cos(2*x), np.sin(2 * x)] for x in coord_x])

Qe = np.mean([loss(w, x, y) for x, y in zip(x_train, coord_y)])
np.random.seed(0)

for _ in range(N):
    k = np.random.randint(0, sz - 1)
    w = w - eta * df(w, x_train[k], coord_y[k])
    ek = loss(w, x_train[k], coord_y[k])
    Qe = lm * ek + (1 - lm) * Qe

Q = np.mean([loss(w, x, y) for x, y in zip(x_train, coord_y)])

print(Q, Qe, w)