import numpy as np

x = np.arange(-2, 3, 0.1)
y = -x + 0.2 * x ** 2 - 0.5 * np.sin(4*x) + np.cos(2*x)

x_train = np.array(x)
y_train = np.array(y)

t = 0

R1 = y[x < 0]
R2 = y[x >= 0]

b1 = np.mean(R1)
b2 = np.mean(R2)

HR1 = np.sum((b1 - R1)**2)
HR2 = np.sum((b2 - R2)**2)

b = np.mean(y_train)
HR = np.sum((b - y_train)**2)

IG = HR - len(R1)/len(y_train) * HR1 - len(R2)/len(y_train) * HR2

print(IG)