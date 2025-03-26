import numpy as np

x = np.arange(-2, 3, 0.1)
y = -x + 0.2 * x ** 2 - 0.5 * np.sin(4*x) + np.cos(2*x)

x_train = np.array(x)
y_train = np.array(y)

b = np.mean(y_train)
HR = np.sum((b - y_train)**2)

th = x[0]
IG = -100

for t in x:
    R1 = y[x < t]
    R2 = y[x >= t]

    b1 = np.mean(R1)
    b2 = np.mean(R2)

    HR1 = np.sum((b1 - R1) ** 2)
    HR2 = np.sum((b2 - R2) ** 2)

    ig = HR - len(R1) / len(y_train) * HR1 - len(R2) / len(y_train) * HR2

    if ig > IG:
        IG = ig
        th = t

print(th, IG)