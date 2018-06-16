# DO IT FUCKING RIGHT THIS TIME
#-CHECK LECTURES IF YOU HAVE TO. ITS BETTER THAN LOOKING AT CODE
#-IF NOT ABLE TO DO IT I VECTOR FROM DO IT IN LOOPS AND THEN IMPROVE IT
#-FUNCKJA KOSZTU J TO SQUARES CZYLI NIE MA ENTROPI JAKOS KOSZTU?

# create a netword with 2 layers instead of standard 1 layer

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 500
X = np.random.random((N, 2)) * 4 - 2  # in between (-2, +2)
Y = X[:, 0] * X[:, 1]

# print(Y)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(Xtrain[:, 0], Xtrain[:, 1], Ytrain)
# plt.show()

# PROBALBY RANDOM DATA

D = 2
M = 100
#K = 1

# initialize weights
W1 = np.random.randn(D, M) / np.sqrt(D)
b1 = np.zeros(M)
W2 = np.random.randn(M) / np.sqrt(M)  # did not work W2 = np.random.randn(N, K) / np.sqrt(D)
b2 = 0  # did not work np.zeros(K)

W = np.random.randn(D, M) / np.sqrt(D)
b = np.zeros(M)

# layer 2
V = np.random.randn(M) / np.sqrt(M)
c = 0


def forward(X):
    Z = X.dot(W) + b
    Z = Z * (Z > 0)  # relu
    #Z = np.tanh(Z)

    Yhat = Z.dot(V) + c
    return Yhat, Z


def derivative_V(Z, Y, Yhat):
    #    return (Yhat - Y).dot(Z)
    return (Y - Yhat).dot(Z)


def derivative_W(X, Z, Y, Yhat, V):
    dZ = np.outer(Y - Yhat, V) * (Z > 0)  # this is for tanh activation
    # dZ = np.outer(Y - Yhat, V) * (Z > 0) # relu
    return X.T.dot(dZ)


def derivative_c(Y, Yhat):
    return (Y - Yhat).sum()


def derivative_b(Z, Y, Yhat, V):
    dZ = np.outer(Y - Yhat, V) * (Z > 0)  # this is for tanh activation
    # dZ = np.outer(Y - Yhat, V) * (Z > 0) # this is for relu activation
    return dZ.sum(axis=0)
# squares


def cost(Y, Yhat):
    return ((Y - Yhat)**2).mean()  # (T - Y).dot(T - Y)  tutaj cos nie dziala


# dobrze


# def classification_rate(T, Y):
#    return np.mean(T == Y)

def update(X, Z, Y, Yhat, W, b, V, c, learning_rate=1e-5):
    gV = derivative_V(Z, Y, Yhat)
    gc = derivative_c(Y, Yhat)
    gW = derivative_W(X, Z, Y, Yhat, V)
    gb = derivative_b(Z, Y, Yhat, V)

    V += learning_rate * gV
    c += learning_rate * gc
    W += learning_rate * gW
    b += learning_rate * gb

    return W, b, V, c


# Xtrain, Ytrain, Xtest, Ytest =


train_costs = []
# test_costs = []
steps = 20000

for epoch in range(steps):
    Yhat, Z = forward(X)  # , W1, W2, b1, b2
    # pYtest = predict(W1, W2, b1, b2, Xtest)
    ctrain = cost(Y, Yhat)
    train_costs.append(ctrain)
    if epoch % 100 == 0:
        pass
        print(ctrain)
        # ctest = cost(pYtest, Ytest)

        # append.test_costs(ctest)

    W, b, V, c = update(X, Z, Y, Yhat, W, b, V, c)


# print("classification rate train:", classification_rate(pYtrain, Ytrain))
# print("classification rate test:", classification_rate(pYtest, Ytest))

# mozna splotowac punkty i spoltowac linie ktora przewiduje ich polozenie

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y)

# surface plot
line = np.linspace(-2, 2, 20)
xx, yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
Yhat, Z = forward(Xgrid)  # , W1, W2, b1, b2
# print(Xgrid[:, 0].shape)
# print(Xgrid[:, 1].shape)
# print(Yhat.shape)

ax.plot_trisurf(Xgrid[:, 0], Xgrid[:, 1], Yhat, linewidth=0.2, antialiased=True)
plt.show()

Ygrid = Xgrid[:, 0] * Xgrid[:, 1]
R = np.abs(Ygrid - Yhat)

# print(Ygrid)

plt.scatter(Xgrid[:, 0], Xgrid[:, 1], c=R)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(Xgrid[:, 0], Xgrid[:, 1], R, linewidth=0.2, antialiased=True)
plt.show()

plt.plot(train_costs, label='train')
# plt.plot(test_costs, label='test')
plt.legend()
plt.show()
