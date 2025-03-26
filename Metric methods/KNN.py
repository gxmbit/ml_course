import numpy as np
from sklearn.model_selection import train_test_split


np.random.seed(0)
n_feature = 5

def distance(x1, x2):
    return np.sum((x1-x2)**2)

def knn(k, x):
    min_distances = []
    for i in range(len(x_train)):
        dist = distance(x, x_train[i])
        if len(min_distances) != k:
            min_distances.append((dist, i, y_train[i]))
        else:
            if min_distances[k-1][0] > dist:
                min_distances[k-1] = (dist, i, y_train[i])
        min_distances.sort(key=lambda d: d[0])
    return min_distances

def to_predict(x):
    info = knn(k, x)
    counter = [info[i][2] for i in range(len(info))]
    prediction = np.argmax([counter.count(0),counter.count(1), counter.count(2)])
    return prediction

r1 = 0.7
D1 = 3.0
mean1 = [3, 7, -2, 4, 6]
V1 = [[D1 * r1 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

r2 = 0.5
D2 = 2.0
mean2 = [3, 7, -2, 4, 6] + np.array(range(1, n_feature+1)) * 0.5
V2 = [[D2 * r2 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

r3 = -0.7
D3 = 1.0
mean3 = [3, 7, -2, 4, 6] + np.array(range(1, n_feature+1)) * -0.5
V3 = [[D3 * r3 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

N1, N2, N3 = 100, 120, 90
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T
x3 = np.random.multivariate_normal(mean3, V3, N3).T

data_x = np.hstack([x1, x2, x3]).T
data_y = np.hstack([np.zeros(N1), np.ones(N2), np.ones(N3) * 2])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123,test_size=0.3, shuffle=True)

k = 5

predict = [to_predict(x) for x in x_test]

Q = np.mean([predict[i] != y_test[i] for i in range(len(x_test))])

print(Q, predict)