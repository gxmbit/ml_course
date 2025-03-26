import numpy as np


def func(x):
    return 0.1 * x - np.cos(x/2) + 0.4 * np.sin(3*x) + 5

def po(x1, x2):
    return np.sum(np.abs(x1 - x2))

def w(x1, x2):
    return 1/np.sqrt(2 * np.pi) * np.exp(-(po(x1, x2) / h)**2/2)


np.random.seed(0)

x = np.arange(-5.0, 5.0, 0.1)
y = func(x) + np.random.normal(0, 0.2, len(x))

x_est = np.arange(-5.0, 5.1, 0.1)

y_est = []
h = 0.5

for xx in x:
    ww = np.array([w(xx, x_) for x_ in x])
    yy = y @ ww/sum(ww)
    y_est.append(yy)

Q = np.mean([(pred - yi)**2 for pred, yi in zip(y_est, y)])

print(Q, y_est)

