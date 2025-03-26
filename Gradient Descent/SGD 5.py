import numpy as np

def func(x):
    return 0.02 * np.exp(-x) - 0.2 * np.sin(3 * x) + 0.5 * np.cos(2 * x) - 7

def s(x):
    return np.array([1, x, x**2, x**3, x**4])

def loss(w, x, y):
    return (w @ s(x) - y) ** 2

def df(w, x, y):
    return 2 * (w @ s(x) - y) * s(x)



coord_x = np.arange(-5.0, 5.0, 0.1)
coord_y = func(coord_x)

sz = len(coord_x)
eta = np.array([0.01, 1e-3, 1e-4, 1e-5, 1e-6])
w = np.array([0., 0., 0., 0., 0.])
N = 500
lm = 0.02

Qe = np.mean([loss(w, x, y) for x, y in zip(coord_x, coord_y)])
np.random.seed(0)

for _ in range(N):
    k = np.random.randint(0, sz - 1)
    ek = loss(w, coord_x[k], coord_y[k])
    w = w - eta * df(w, coord_x[k], coord_y[k])
    Qe = lm * ek + (1 - lm) * Qe

Q = np.mean([loss(w, x, y) for x, y in zip(coord_x, coord_y)])

print(w, Q, Qe, sep='\n')