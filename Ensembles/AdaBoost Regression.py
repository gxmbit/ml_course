import numpy as np
from sklearn.tree import DecisionTreeRegressor


x = np.arange(-3, 3, 0.1).reshape(-1, 1)
y = 2 * np.cos(x) + 0.5 * np.sin(2*x) - 0.2 * np.sin(4*x)

T = 6
S = np.array(y.ravel())
algs = []

for _ in range(T):
    model = DecisionTreeRegressor(max_depth=3)

    model.fit(x, S)

    algs.append(model)

    S -= model.predict(x)

ax = lambda xx: np.sum([a.predict(xx.reshape(1, -1)) for a in algs])

predict = [ax(xx) for xx in x]

QT = np.mean((y.ravel() - predict)**2)

print(y)