import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []

for line in open('data_poly.csv'):
    x, y = line.split(',')

    x = float(x)

    X.append([1, x, x * x])
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

#plt.scatter(X, Y)
# plt.show()

w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
print(w)

Y_hat = np.dot(X, w)

# r2

d1 = Y - Y_hat
d2 = Y - Y.mean()

r2 = 1 - d1.dot(d1) / d2.dot(d2)

print(r2)


plt.scatter(X[:, 1], Y)
plt.plot(sorted(X[:, 1]), sorted(Y_hat))
plt.show()
