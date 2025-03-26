import numpy as np


def func(x):
    return 2 * x + 0.1 * x ** 3 + 2 * np.cos(3*x)

def df(x):
    return 2 + 0.3 * x**2 - 6 * np.sin(3*x)

n = 0.5
x0 = 4.0
N = 200
alpha = 0.8
G = 0
ep = 0.01
x = x0

for _ in range(N):
    G = alpha * G + (1 - alpha) * df(x)**2
    x = x - n * df(x)/(np.sqrt(G) + ep)

print(x)