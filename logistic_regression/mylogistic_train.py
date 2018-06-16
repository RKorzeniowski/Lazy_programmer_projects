from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from process import get_binary_data

# get the data
X, Y = get_binary_data()

# withold 100 samples as test set
Xtrain = X[:-100]
Ytrain = Y[:-100]
Xtest = X[-100:]
Ytest = Y[-100:]

D = X.shape[1]  # size D
W = np.random.randn(D)
b = 0

# print(X.shape[1])


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def forward(X, W, b):
    return sigmoid(X.dot(W) + b)

# calculate the accuracy


def classificaiton_rate(Y, P):
    return np.mean(Y == P)

# cross entropy


def cross_entrpy_error(T, pY):
    return -np.mean(T * np.log(pY) + (1 - T) * np.log(1 - pY))


w = np.random.randn(D)
w1 = w

train_cost = []
test_cost = []
lr = 0.001
epochs = 10000
l2 = 1

for i in range(epochs):
    pYtrain = forward(Xtrain, w, b)
    pYtest = forward(Xtest, w, b)

    ctrain = cross_entrpy_error(Ytrain, pYtrain)
    ctest = cross_entrpy_error(Ytest, pYtest)
    train_cost.append(ctrain)
    test_cost.append(ctest)

    # gradient descent
    w -= lr * np.dot(Xtrain.T, (pYtrain - Ytrain))
    b -= lr * (pYtrain - Ytrain).sum()
    if i % 1000 == 0:
        print(i, ctrain, ctest)

for i in range(epochs):
    pYtrain = forward(Xtrain, w1, b)
    pYtest = forward(Xtest, w1, b)

    ctrain = cross_entrpy_error(Ytrain, pYtrain)
    ctest = cross_entrpy_error(Ytest, pYtest)
    train_cost.append(ctrain)
    test_cost.append(ctest)

    # gradient descent
    w1 -= lr * (Xtrain.T.dot(pYtrain - Ytrain) + l2 * w1)
    b -= lr * ((pYtrain - Ytrain).sum())
    if i % 1000 == 0:
        print(i, ctrain, ctest)


print("final classification train ", classificaiton_rate(Ytrain, np.round(pYtrain)))
print("final classification test ", classificaiton_rate(Ytest, np.round(pYtest)))


# plot train cost

# for i in range(epochs):
#     Y_pred = np.dot(Xtrain, w1.T)
#     w1 += lr * np.dot(Xtrain.T, (Ytrain - Y_pred))
#     train_cost.append(cross_entrpy_error(Ytrain, Y_pred))

print("w", w)
print("w1", w1)
# make prediciton on testset


legend1, = plt.plot(train_cost, label='training cost')
legend2, = plt.plot(test_cost, label='test cost')
plt.legend([legend1, legend2])
plt.show()
