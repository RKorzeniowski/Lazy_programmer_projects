import numpy as np
import matplotlib.pyplot as plt
N = 1000
D = 2

R_inner = 5
R_outer = 10

print(N / 2)

R1 = np.random.randn(N // 2) + R_inner
theta = 2 * np.pi * np.random.random(N // 2)
X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

R2 = np.random.randn(N // 2) + R_outer
theta = 2 * np.pi * np.random.random(N // 2)
X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

X = np.concatenate([X_inner, X_outer])
T = np.array([0] * (N // 2) + [1] * (N // 2))

#plt.scatter(X[:, 0], X[:, 1], c=T)
# plt.show()

# column of ones for bias term
ones = np.array([[1] * N]).T

# column which reprezents radious of a point
r = np.zeros((N, 1))
for i in range(N):
    r[i] = np.sqrt(X[i, :].dot(X[i, :]))  # manulay calculate the radious

Xb = np.concatenate((ones, r, X), axis=1)

w = np.random.randn(D + 2)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


z = Xb.dot(w)

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
lr = 0.0001
error = []
for i in range(5000):
    e = cross_entropy(T, Y)
    error.append(e)
    if i % 100 == 0:
        print(e)
    w += lr * (np.dot((T - Y).T, Xb) - l2 * w)
    Y = sigmoid(np.dot(Xb, w))

plt.plot(error)
plt.title("cross entropy")
plt.show()

print("final dablju,w", w)

print("final classification rate", 1 - np.abs(T - np.round(Y)).sum() / N)
