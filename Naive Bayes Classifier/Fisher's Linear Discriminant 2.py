import numpy as np

np.random.seed(0)

mean1 = np.array([1, -2, 0])
mean2 = np.array([1, 3, 1])
r = 0.7
D = 2.0
V = [[D, D * r, D*r*r], [D*r, D, D*r], [D*r*r, D*r, D]]

N = 1000
x1 = np.random.multivariate_normal(mean1, V, N).T
x2 = np.random.multivariate_normal(mean2, V, N).T

x_train = np.hstack([x1, x2]).T
y_train = np.hstack([np.zeros(N), np.ones(N)])

mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

V1 = np.cov(x1)
V2 = np.cov(x2)
Sigma = (N * V1 + N * V2) / (2 * N)

Py1, L1 = 0.5, 1
Py2, L2 = 1 - Py1, 1

alpha1 = np.linalg.inv(Sigma) @ mm1
alpha2 = np.linalg.inv(Sigma) @ mm2

beta1 = np.log(L1*Py1) - 0.5 * mm1 @ np.linalg.inv(Sigma) @ mm1
beta2 = np.log(L2*Py2) - 0.5 * mm2 @ np.linalg.inv(Sigma) @ mm2

print(alpha1, alpha2, beta1, beta2, sep='\n')