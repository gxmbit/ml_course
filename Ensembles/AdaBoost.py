import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


np.random.seed(0)
n_feature = 2

r1 = 0.7
D1 = 3.0
mean1 = [3, 7]
V1 = [[D1 * r1 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

r2 = 0.5
D2 = 2.0
mean2 = [4, 2]
V2 = [[D2 * r2 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

N1, N2 = 1000, 1200
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N1) * -1, np.ones(N2)])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123,test_size=0.3, shuffle=True)

max_depth = 3
T = 10
w = np.ones(len(x_train)) / len(x_train)
algs = []
alfas = []

for i in range(T):
    algs.append(DecisionTreeClassifier(criterion="gini", max_depth=max_depth))

    algs[i].fit(x_train, y_train, sample_weight=w)

    prediction = algs[i].predict(x_train)
    N = np.sum(np.abs(y_train - prediction) / 2 * w)
    alfas.append(0.5 * np.log((1 - N) / N) if N != 0 else np.log((1 - 1e-8) / 1e-8))

    w = w * np.exp((-1) * alfas[i] * y_train * prediction)
    w = w/np.sum(w)

ax = lambda x: np.sign(np.sum([alfas[i] * algs[i].predict(x.reshape(1, -1)) for i in range(T)]))

predict = [ax(x) for x in x_test]

Q = np.sum(predict != y_test)

print(Q)


