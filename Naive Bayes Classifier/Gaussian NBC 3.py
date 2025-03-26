import numpy as np

np.random.seed(0)

r1 = 0.7
D1 = 1.0
mean1 = [-1, -2, -1]
V1 = [[D1, D1 * r1, D1*r1*r1], [D1 * r1, D1, D1*r1], [D1*r1*r1, D1*r1, D1]]

r2 = 0.5
D2 = 2.0
mean2 = [1, 2, 1]
V2 = [[D2, D2 * r2, D2*r2*r2], [D2 * r2, D2, D2*r2], [D2*r2*r2, D2*r2, D2]]

N = 1000
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T

x_train = np.hstack([x1, x2]).T
y_train = np.hstack([np.ones(N) * -1, np.ones(N)])

mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

VV1 = np.cov(x1)
VV2 = np.cov(x2)

Py1, L1 = 0.5, 1
Py2, L2 = 1 - Py1, 1

b = lambda x, v, m, l, py: np.log(l * py) - 0.5 * (x - m) @ np.linalg.inv(v) @ (x - m).T - 0.5 * np.log(np.linalg.det(v))

predict = np.array([np.argmax([b(x, VV1, mm1, L1, Py1), b(x, VV2, mm2, L2, Py2)]) * 2 - 1 for x in x_train])

Q = sum([1 if predict[i] != y_train[i] else 0 for i in range(len(y_train))])

print(predict, Q)