import numpy as np
import matplotlib.pyplot as plt

N = 50
D = 50

# uniform distribution point in matrix (NxD), around 0 and value -5 to +5
X = (np.random.random((N, D)) - 0.5) * 10

true_w = np.array([1, 0.5, -0.5] + [0] * (D - 3))

Y = X.dot(true_w) + np.random.randn(N) * 0.5

# print(X)
# print(Y)

# gotta add outliners

# random initializaon of weights
w = np.random.randn(D) / np.sqrt(D)
lr = 0.001
costs = []
epochs = 1000
l1 = 10
# max likelihood prob
for i in range(epochs):
    Y_hat = X.dot(w)
    delta = Y_hat - Y
    w = w - lr * (l1 * np.sign(w) + X.T.dot(delta))
    mse = delta.dot(delta) / N
    costs.append(mse)


plt.plot(costs)
plt.show()

plt.plot(true_w, label='list of tru dablju(orginal weights)')
plt.plot(w, label='the pretendor')
plt.legend()
plt.show()
