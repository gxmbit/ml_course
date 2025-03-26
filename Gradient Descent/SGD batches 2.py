import numpy as np

def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 2.5

def s(x):
    return np.array([1, x, x**2, x**3])

def error(x, y):
    return w @ x - y

coord_x = np.arange(-4.0, 6.0, 0.1)
coord_y = func(coord_x)

sz = len(coord_x)
eta = np.array([0.1, 0.01, 0.001, 0.0001])
w = np.array([0., 0., 0., 0.])
N = 500
lm = 0.02
batch_size = 50

Qe = 1/sz * sum([(w @ s(x) - y)**2 for x, y in zip(coord_x, coord_y)])
np.random.seed(0)

for _ in range(N):
    k = np.random.randint(0, sz - batch_size)
    Qk = 1 / batch_size * sum([error(s(x), y) ** 2 for x, y in zip(coord_x[k: k + batch_size], coord_y[k: k + batch_size])])
    w = w - eta * 2 / batch_size * sum([error(s(x), y) * s(x) for x, y in zip(coord_x[k: k + batch_size], coord_y[k: k + batch_size])])
    Qe = lm*Qk + (1-lm)*Qe

Q = 1/sz * sum([error(s(x), y) ** 2 for x, y in zip(coord_x, coord_y)])

print(w, Q, Qe, sep='\n')