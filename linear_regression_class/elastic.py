import numpy as np
import matplotlib.pyplot as plt

# create fat matrix with too many features
N = 50
D = 50

Xb = (np.random.random((N, D)) - 0.5) * 10

# jak zrobic zeby dodac outlinersow do datasetu (to co jest w przeciwdzialane przez L2)
# czemu wyrzuca error o Nan wartosciach jak tylko zwiekszylem N z 10 na 50
# for i in range(N):
#     print(X[i, 1])
#     X[i, 1] = X[i, 1] * 3
#print(X[:, 1])
# print(X)

outliners = np.array([[np.random.random() * 10] * N]).T
X = np.concatenate((outliners, Xb), axis=1)


print(X)

#print(X[N - 1, :])

# add some outliers
#print(X[N - 1, :])

w_tru = np.random.random(D)  # to by bylo wyzwanie od 0 do 1 totaly rmd weights
true_w = np.array([1, 0.5, -0.5] + [0] * (D - 2))


# plt.plot(true_w, label='list of tru dablju(orginal weights)')
# plt.plot(w_tru, label='the pretendor')
# plt.legend()
# plt.show()


Y = X.dot(true_w) + np.random.randn(N) * 0.5

# print(Y)


# grad with elastic => l1+l2

lr = 0.001
costs = []
epochs = 1000
l1 = 10
l2 = 10
# normlay distributed with variance of 1/D
w = np.random.randn(D + 1) / np.sqrt(D + 1)
ws = w
for i in range(epochs):
    Y_hat = np.dot(X, ws)
    delta = Y_hat - Y
    ws = ws - lr * (l1 * np.sign(ws) + l2 * ws + X.T.dot(delta))
    mse = delta.dot(delta) / N
    costs.append(mse)


w1 = w
lr = 0.001
costs2 = []
epoch2 = 1000
l1 = 10
# max likelihood prob
for i in range(epoch2):
    Y_hat1 = X.dot(w1)
    delta2 = Y_hat1 - Y
    w1 = w1 - lr * (l1 * np.sign(w1) + X.T.dot(delta2))
    mse1 = delta2.dot(delta2) / N
    costs2.append(mse1)


plt.plot(costs, label='l1+l2')
plt.plot(costs2, label='l1')
plt.legend()
plt.show()

plt.plot(true_w, label='list of tru dablju(orginal weights)')
plt.plot(ws, label='the pretendor l1+l2')
plt.plot(w1, label='the pretendor l1')
plt.legend()
plt.show()
