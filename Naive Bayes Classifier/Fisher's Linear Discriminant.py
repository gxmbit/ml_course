import numpy as np

np.random.seed(0)

mean1 = [1, -2]
mean2 = [1, 3]
r = 0.7
D = 2.0
V = [[D, D * r], [D * r, D]]

N = 1000
x1 = np.random.multivariate_normal(mean1, V, N).T
x2 = np.random.multivariate_normal(mean2, V, N).T

x_train = np.hstack([x1, x2]).T
y_train = np.hstack([np.ones(N) * -1, np.ones(N)])

mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

Py1, L1 = 0.5, 1
Py2, L2 = 1 - Py1, 1

a = np.hstack([(x1.T - mm1).T, (x2.T - mm2).T])
VV = np.array([[np.dot(a[0], a[0]) / (2*N), np.dot(a[0], a[1]) / (2*N)],
                [np.dot(a[1], a[0]) / (2*N), np.dot(a[1], a[1]) / (2*N)]])

b = lambda l, py, mm, x: np.log(l*py) - 0.5 * mm @ np.linalg.inv(VV) @ mm + x @ np.linalg.inv(VV) @ mm

predict = np.array([np.argmax([b(L1, Py1, mm1, x), b(L2, Py2, mm2, x)]) * 2 - 1 for x in x_train])

Q = sum([1 if predict[i] != y_train[i] else 0 for i in range(len(y_train))])

print(predict, Q)