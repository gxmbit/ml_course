import numpy as np

def func(x):
    return -0.5 * x ** 2 + 0.1 * x ** 3 + np.cos(3 * x) + 7

def model(w, x):
    xv = np.array([x ** n for n in range(len(w))])
    return w.T @ xv


def loss(w, x, y):
    return (model(w, x) - y) ** 2

def dL(w, x, y):
    xv = np.array([x ** n for n in range(len(w))])
    return 2 * (model(w, x) - y) * xv


coord_x = np.arange(-4.0, 6.0, 0.1)
coord_y = func(coord_x)

N = 5
lm_l1 = 2.0
sz = len(coord_x)
eta = np.array([0.1, 0.01, 0.001, 0.0001, 0.000002])
w = np.zeros(N)
n_iter = 500
lm = 0.02
batch_size = 20

Qe = np.mean([loss(w, x, y) for x,y in zip(coord_x, coord_y)])
np.random.seed(0)

for _ in range(n_iter):
    k = np.random.randint(0, sz - batch_size - 1)
    batch_x = coord_x[k:k + batch_size]
    batch_y = coord_y[k:k + batch_size]
    ek = np.mean([loss(w, x, y) for x, y in zip(batch_x, batch_y)])
    w = w - eta * (np.mean([dL(w, x, y) for x,y in zip(batch_x, batch_y)], axis=0) + lm_l1 * np.sign(np.concatenate(([0], w[1:]))))
    Qe = lm * ek + (1 - lm) * Qe

Q = np.mean([loss(w, x, y) for x,y in zip(coord_x, coord_y)])

print(Q, Qe, w, sep='\n')