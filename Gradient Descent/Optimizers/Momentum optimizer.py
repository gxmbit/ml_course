import numpy as np


def func(x):
    return -0.5 * x + 0.2 * x ** 2 - 0.01 * x ** 3 - 0.3 * np.sin(4*x)

def df(x):
    return -0.5 + 0.4 * x - 0.03 * x**2 - 1.2 * np.cos(4*x)

n = 0.1
x0 = -3.5
N = 200
eta = 0.8
v = 0
x = x0

for _ in range(N):
    v = eta*v + (1 - eta) * n * df(x)
    x = x - v

print(v)
