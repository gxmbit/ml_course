import numpy as np

np.random.seed(0)
X = np.random.randint(0, 2, size=200)

X1 = X[:150]
X2 = X[150:]

def S(x):
    return 1 - ((x[x == 0].size/x.size)**2 + (x[x == 1].size/x.size)**2)

IG = S(X) - (3/4 * S(X1) + 1/4 * S(X2))

print(IG)