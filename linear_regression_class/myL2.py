from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import matplotlib.pyplot as plt

N = 50

# generate the data
X = np.linspace(0, 10, N)
Y = 0.5 * X + np.random.randn(N)

# make outliers
Y[-1] += 30
Y[-2] += 30


# plot the data
# plt.scatter(X, Y)
# plt.show()

# X0 = np.linspace(1, 1, N)
# better


print(X)
print(Y)

X = np.vstack([np.ones(N), X]).T
Y = np.asarray(Y)

# maximum likelihood
w_ml = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))

Y_hat = np.dot(X, w_ml)

# plt.scatter(X[:, 1], Y)

# plt.show()

d1 = Y - Y_hat
d2 = Y - Y.mean()

r2 = 1 - d1.dot(d1) / d2.dot(d2)
print(r2)

# L2


# lambda.dot(w)

l2 = 1000

w_map = np.linalg.solve(l2 * np.eye(2) + np.dot(X.T, X), np.dot(X.T, Y))

Y_hat_map = np.dot(X, w_map)

d1 = Y - Y_hat_map

r2_l = 1 - d1.dot(d1) / d2.dot(d2)
print(r2_l)

plt.scatter(X[:, 1], Y)
plt.plot(X[:, 1], Y_hat, label="max likelihood")
plt.plot(X[:, 1], Y_hat_map, label="map")
plt.legend()
plt.show()
