import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split


np.random.seed(0)

r1 = 0.6
D1 = 3.0
mean1 = [1, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.7
D2 = 2.0
mean2 = [-3, -1]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

r3 = 0.5
D3 = 1.0
mean3 = [1, 2]
V3 = [[D3, D3 * r3], [D3 * r3, D3]]

N = 500
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T
x3 = np.random.multivariate_normal(mean3, V3, N).T

data_x = np.hstack([x1, x2, x3]).T
data_y = np.hstack([np.zeros(N), np.ones(N), np.ones(N) * 2])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123,test_size=0.3, shuffle=True)

clf = svm.SVC(kernel='linear')

clf.fit(x_train, y_train)

w1 = clf.coef_[0]
w2 = clf.coef_[1]
w3 = clf.coef_[2]

w01 = clf.intercept_[0]
w02 = clf.intercept_[1]
w03 = clf.intercept_[2]

w1 = np.hstack((w01,w1))
w2 = np.hstack((w02,w2))
w3 = np.hstack((w03,w3))

predict = clf.predict(x_test)

Q = sum([1 if y != predicted else 0 for y, predicted in zip(y_test, predict)])

print(w1, w2, w3, Q, sep='\n')