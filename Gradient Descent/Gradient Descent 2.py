import numpy as np


def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.1 * x ** 3


def df(x):
    return 0.5 + 0.4 * x - 0.3 * x ** 2


coord_x = np.arange(-5.0, 5.0, 0.1)
coord_y = func(coord_x)

eta = 0.01
x = -4
N = 200

for _ in range(N):
    x = x - eta * df(x)