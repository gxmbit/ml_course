import numpy as np

def func(x):
    return 0.5 * x**2 - 0.1 * 1/np.exp(-x) + 0.5 * np.cos(2*x) - 2.

def s(x):
    return np.array([1, x, x**2, np.cos(2*x), np.sin(2*x)])

coord_x = np.arange(-5.0, 5.0, 0.1)
coord_y = func(coord_x)

sz = len(coord_x)
eta = np.array([0.01, 0.001, 0.0001, 0.01, 0.01])
w = np.array([0., 0., 0., 0., 0.])
N = 500
lm = 0.02

Qe = 1/sz * sum([(w @ s(x) - y)**2 for x, y in zip(coord_x, coord_y)])

for _ in range(N):
    k = np.random.randint(0, sz)
    ek = (w @ s(coord_x[k]) - coord_y[k])**2
    w = w - eta * 2 * (w @ s(coord_x[k]) - coord_y[k]) * s(coord_x[k]).T
    Qe = lm * ek + (1 - lm) * Qe

Q = 1/sz * sum([(w @ s(x) - y)**2 for x, y in zip(coord_x, coord_y)])

print(w, Q, Qe, sep='\n')
