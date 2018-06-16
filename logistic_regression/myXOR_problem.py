import numpy as np
import matplotlib.pyplot as plt

N = 4
D = 2

# XOR
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])
T = np.array([0, 1, 1, 0])

# add a column of ones
# ones = np.array([[1]*N]).T
ones = np.ones((N, 1))


#plt.scatter(X[:, 0], X[:, 1], c=T)
# plt.show()

# add a column of xy = x*y
xy = (X[:, 0] * X[:, 1]).reshape(N, 1)
Xb = np.concatenate((ones, xy, X), axis=1)


w = np.random.randn(D + 2)

z = Xb.dot(w)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


Y = sigmoid(z)

# calculate the cross-entropy error


def cross_entropy(T, Y):
    E = 0
    for i in range(len(T)):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E


l2 = 0.01
lr = 0.001
error = []
for i in range(10000):
    e = cross_entropy(T, Y)
    error.append(e)
    if i % 1000 == 0:
        print(e)

    # gradient descent weight udpate with regularization
    w += lr * (Xb.T.dot(T - Y) - l2 * w)

    # recalculate Y
    Y = sigmoid(Xb.dot(w))

plt.plot(error)
plt.title("cross entropy")
plt.show()

print("final dablju,w", w)

print("final classification rate", 1 - np.abs(T - np.round(Y)).sum() / N)