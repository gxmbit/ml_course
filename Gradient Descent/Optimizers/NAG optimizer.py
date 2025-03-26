import numpy as np


def func(x):
    return 0.4 * x + 0.1 * np.sin(2*x) + 0.2 * np.cos(3*x)

def df(x):
    return 0.4 + 0.2 * np.cos(2*x) - 0.6 * np.sin(3*x)

n = 1.0
x0 = 4
N = 500
eta = 0.7
v = 0
x = x0

for _ in range(N):
    v = eta * v + (1 - eta) * df(x - eta * v)
    x = x - v

print(x)