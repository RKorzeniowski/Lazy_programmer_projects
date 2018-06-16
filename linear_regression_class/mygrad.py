import numpy as np
import matplotlib.pyplot as plt

N = 10
D = 3

X = np.zeros((N, D))

X[:, 0] = 1

X[:5, 1] = 1

X[5:, 2] = 1

print(X)

Y = np.array([0] * 5 + [1] * 5)

print(Y)

# ling reg does not work
# w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))  numpy.linalg.linalg.LinAlgError: Singular matrix

# grad desent
l_rate = 0.001

# w = np.random.normal(0, 1 / D) same as
w = np.random.randn(D) / np.sqrt(D)


costs = []
epoch = 1000
for i in range(epoch):
    Y_hat = X.dot(w)
    delta = Y_hat - Y
    w = w - l_rate * X.T.dot(delta)
    # mean squared error
    mse = delta.dot(delta) / N
    costs.append(mse)
    print(mse)


print(w)
plt.plot(costs)
plt.show()


plt.plot(Y_hat, label='predictions')
plt.plot(Y, label='data')
plt.legend()
plt.show()
