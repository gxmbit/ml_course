import numpy as np

np.random.seed(0)

mean1 = np.array([1, -2])
mean2 = np.array([-3, -1])
mean3 = np.array([1, 2])

r = 0.5
D = 1.0
V = [[D, D * r], [D*r, D]]

N = 1000
x1 = np.random.multivariate_normal(mean1, V, N).T
x2 = np.random.multivariate_normal(mean2, V, N).T
x3 = np.random.multivariate_normal(mean3, V, N).T

x_train = np.hstack([x1, x2, x3]).T
y_train = np.hstack([np.zeros(N), np.ones(N), np.ones(N) * 2])

mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)
mm3 = np.mean(x3.T, axis=0)

V1 = np.cov(x1)
V2 = np.cov(x2)
V3 = np.cov(x3)

Sigma = (N * V1 + N * V2 + N*V3) / (3 * N)
Sigma_inv = np.linalg.inv(Sigma)

Py1, Py2, Py3 = 0.2, 0.4, 0.4
L1, L2, L3 = 1, 1, 1

alpha = lambda mm: Sigma_inv @ mm
beta = lambda L, Py, mm: np.log(L * Py) - 0.5 * mm @ Sigma_inv @ mm

a = lambda x, mm, L, Py: x @ alpha(mm) + beta(L, Py, mm)

predict = np.array([np.argmax([a(x, mm1, L1, Py1), a(x, mm2, L2, Py2), a(x, mm3, L3, Py3)]) for x in x_train])

Q = sum(predict != y_train)

print(predict, Q)