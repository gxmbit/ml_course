import numpy as np

np.random.seed(0)

n_total = 1000
n_features = 200

table = np.zeros(shape=(n_total, n_features))

for _ in range(100):
    i, j = np.random.randint(0, n_total), np.random.randint(0, n_features)
    table[i, j] = np.random.randint(1, 10)

F = 1/len(table) * table.T @ table
L, W = np.linalg.eig(F)
WW = sorted(zip(L, W), key = lambda x: x[0], reverse=True)
WW = np.array([W[1] for W in WW])
data_x = table @ WW.T
data_x = data_x[:, :52]
print(len(L[L <= 0.01]), len(L), data_x.shape)