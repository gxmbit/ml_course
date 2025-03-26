import numpy as np

np.random.seed(0)

r = 0.7
D = 3.0
mean = [3, 7, -2, 4, 6]
n_feature = 5
V = [[D * r ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

N = 1000
X = np.random.multivariate_normal(mean, V, N).T

F = 1/X.shape[1] * X @ X.T

L, W = np.linalg.eig(F)