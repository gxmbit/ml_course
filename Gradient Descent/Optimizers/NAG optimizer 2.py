import numpy as np

def func(x):
    return -0.7 * x - 0.2 * x ** 2 + 0.05 * x ** 3 - 0.2 * np.cos(3 * x) + 2

def s(x):
    return np.array([1, x, x**2, x**3])

coord_x = np.arange(-4.0, 6.0, 0.1)
coord_y = func(coord_x)

sz = len(coord_x)
eta = np.array([0.1, 0.01, 0.001, 0.0001])
w = np.array([0., 0., 0., 0.])
N = 500
lm = 0.02
batch_size = 20
gamma = 0.8
v = np.zeros(len(w))

Qe = np.mean([(w @ s(coord_x[i]) - coord_y[i])**2 for i in range(sz)])
np.random.seed(0)

for _ in range(N):
    k = np.random.randint(0, sz - batch_size - 1)
    Qk = np.mean([(w @ s(coord_x[k]) - coord_y[k])**2 for k in range(k, k + batch_size)])
    v = gamma * v + (1 - gamma) * eta * 2 / batch_size * sum([((w - gamma*v).T @ s(coord_x[k]) - coord_y[k]) * s(coord_x[k]) for k in range(k, k+batch_size)])
    w = w - v
    Qe = lm * Qk + (1 - lm) * Qe

Q = np.mean([(w @ s(coord_x[i]) - coord_y[i])**2 for i in range(sz)])

print(Q, Qe, w, sep='\n')