import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2


X = np.random.randn(N, D)

# center the first 50 points at (-2,-2)
X[:50, :] = X[:50, :] - 2 * np.ones((50, D))

# center the last 50 points at (2, 2)
X[50:, :] = X[50:, :] + 2 * np.ones((50, D))

T = np.array([0] * 50 + [1] * 50)


# ones = np.array([[1]*N]).T  # tranpose to be Nx1 (created as initialy as (1xN))
ones = np.ones((N, 1))
Xb = np.concatenate((ones, X), axis=1)

# init rmd weights
#w = np.random.randn(D + 1)
w = np.array([0, 4, 4])
# create pred
z = Xb.dot(w)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# pred
Y = sigmoid(z)


def cross_entropy_error(T, Y):
    E = 0
    E1 = 0
    # chyba napisanie tego w postaci calego wzoru daloby taki sam efekt(sprawdzic)
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])  # Y to wektor Nx1
        else:  # T[i] = 0
            E -= np.log(1 - Y[i])
    return E


# chyba powininem zastosowac siga na koniec
lr = 0.1
epochs = 100
errors = []
errors2 = []

l2 = 0.1
w = np.random.randn(D + 1)
w1 = w
w2 = w

for i in range(epochs):
    # if i % 10 == 0:
    Y_hat = np.dot(Xb, w)
    Y_hat2 = np.dot(Xb, w)
    w += lr * (np.dot((T - Y).T, Xb) - l2 * w)
    w1 -= lr * np.dot((Y - T).T, Xb)  # looks like it does
    Y = sigmoid(Y_hat)
    Y2 = sigmoid(Y_hat2)
    errors.append(cross_entropy_error(T, Y))
    errors2.append(cross_entropy_error(T, Y2))


print(errors)
print("w", w)
print("w1", w1)

plt.plot(errors)
plt.show()


learning_rate = 0.1

for i in range(epochs):
    if i % 10 == 0:
        pass
        #print(cross_entropy_error(T, Y))
    w2 += learning_rate * np.dot((T - Y).T, Xb)  # X.T.dot(T-Y)
    Y = sigmoid(Xb.dot(w2))

print("final weight", w2)
