import numpy as np

def po(x1, x2):
    return np.sum(np.abs(x1 - x2))

def w(x1, x2, h):
    return np.abs(1 - po(x1, x2)/h) * bool(np.abs(po(x1, x2)/h) <= 1)

x = np.array([0, 1, 2, 3])
y = np.array([0.5, 0.8, 0.6, 0.2])

x_est = np.arange(0, 3.1, 0.1)

y_est = []

for xx in x_est:
    ww = np.array([w(xx, xi, 1) for xi in x])
    yy = np.dot(ww, y) / sum(ww)
    y_est.append(yy)

print(y_est)

