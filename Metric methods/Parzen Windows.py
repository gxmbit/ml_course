import numpy as np
from sklearn.model_selection import train_test_split


np.random.seed(0)
n_feature = 2



def p(x_1, x_2):
    return np.sum(np.abs(x_1 - x_2))

def K(x_1, x_2):
    return 1/np.sqrt(2 * np.pi) * np.exp(-p(x_1, x_2) ** 2 / 2)

def class_prediction(class_number, x):
    return np.sum([(y == class_number) * K(x, train_x) for train_x, y in zip(x_train, y_train)])

def a(x):
    return np.argmax([class_prediction(0, x), class_prediction(1, x), class_prediction(2, x)])

r1 = 0.7
D1 = 3.0
mean1 = [3, 3]
V1 = [[D1 * r1 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

r2 = 0.5
D2 = 2.0
mean2 = [1, 1]
V2 = [[D2 * r2 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

r3 = -0.7
D3 = 1.0
mean3 = [-2, -2]
V3 = [[D3 * r3 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

N1, N2, N3 = 200, 150, 190
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T
x3 = np.random.multivariate_normal(mean3, V3, N3).T

data_x = np.hstack([x1, x2, x3]).T
data_y = np.hstack([np.zeros(N1), np.ones(N2), np.ones(N3) * 2])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123,test_size=0.5, shuffle=True)

predict = [a(x) for x in x_test]

Q = np.mean(predict != y_test)