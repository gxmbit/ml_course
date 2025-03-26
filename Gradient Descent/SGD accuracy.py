import numpy as np
from sklearn.model_selection import train_test_split


def loss(w, x, y):
    M = w @ x * y
    return np.exp(-M)


def df(w, x, y):
    M = w @ x * y
    return -np.exp(-M) * x * y


np.random.seed(0)

r1 = 0.4
D1 = 2.0
mean1 = [1, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 3.0
mean2 = [2, 3]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

N = 1000
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T

data_x = np.array([[1, x[0], x[1]] for x in np.hstack([x1, x2]).T])
data_y = np.hstack([np.ones(N) * -1, np.ones(N)])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123,test_size=0.3, shuffle=True)

n_train = len(x_train)
w = np.array([0.0, 0.0, 0.0])
nt = np.array([0.5, 0.01, 0.01])
lm = 0.01
N = 500
batch_size = 10

for _ in range(N):
    k = np.random.randint(0, n_train - batch_size - 1)
    x_batch = x_train[k:k+batch_size]
    y_batch = y_train[k:k + batch_size]
    w = w - nt * np.mean([df(w, x, y) for x, y in zip(x_batch, y_batch)], axis = 0)

mrgs = np.array([w @ x * y for x, y in zip(x_test, y_test)])

mrgs.sort()

acc = np.mean([np.sign(w @ x) == y for x, y in zip(x_test, y_test)])

print(w, acc)
