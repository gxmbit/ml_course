import numpy as np

rub_usd = np.array([75, 76, 79, 82, 85, 81, 83, 86, 87, 85, 83, 80, 77, 79, 78, 81, 84])

x = np.array(range(len(rub_usd)))
x_est = np.array(range(len(rub_usd), len(rub_usd) + 10))


h = 3
ro = lambda xi, x: np.abs(xi - x)
K = lambda r: np.exp(-0.5 * r * r) / np.sqrt(2 * np.pi)
w = lambda xx, xi: K(ro(xx, xi) / h)

predict = []

for xx in x_est:
    ww = np.array([w(xx, xi) for xi in x])
    yy = np.dot(ww, rub_usd) / sum(ww)
    predict.append(yy)
    rub_usd = np.append(rub_usd, yy)[1:]
    x = np.append(x, xx)[1:]

print(x)




