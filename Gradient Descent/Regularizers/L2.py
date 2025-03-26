import numpy as np

def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 2.5


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
lm_l2 = 2
sz = len(coord_x)
eta = np.array([0.1, 0.01, 0.001, 0.0001, 0.000002])
w = np.zeros(N)
n_iter = 500
lm = 0.02
batch_size = 20

Qe = np.mean([loss(w, x, y) for x, y in zip(coord_x, coord_y)])
np.random.seed(0)

X = np.array([[a ** n for n in range(N)] for a in coord_x])

for _ in range(n_iter):
    k = np.random.randint(0, sz - batch_size - 1)
    ek = np.mean([loss(w, coord_x[i], coord_y[i]) for i in range(k, k+batch_size)])
    w = w - eta * (np.mean([dL(w, coord_x[i], coord_y[i]) for i in range(k, k + batch_size)], axis=0) + lm_l2 * np.concatenate(([0], w[1:])))
    Qe = lm * ek + (1 - lm) * Qe

Q = np.mean([loss(w, x, y) for x, y in zip(coord_x, coord_y)])

print(Q, Qe, w, sep='\n')