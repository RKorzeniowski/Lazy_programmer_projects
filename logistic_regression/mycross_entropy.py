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
    # taki sam wynik
    for i in range(N):
        E1 -= T[i] * np.log(Y[i]) + (1 - T[i]) * np.log(1 - Y[i])

    # za duzy wynik 100 razy (cos powinno byc transponowane chyba)
    #E2 = -np.dot((T, np.log(Y)) + np.dot( (ones - T), np.log(ones - Y)))

    print("E1", E1)
    #print("E2", np.sum(E2))
    return E

    # sprawdzic czy dziala tak samo [i czy jako wektory w calosci tez dziala ]
    #E = T[i] * np.log(Y[i]) + (1 - T[i]) * np.log(1 - Y[i])


print("just random", cross_entropy_error(T, Y))


# # test log solution, use logistic reggresion to do that

# why linear regression like here does not work (solution for squared error not even good function solved here) so we have to use gradient decent
# w1 = np.linalg.solve(np.dot(Xb.T, Xb), np.dot(Xb.T, T))  # lin reg work meh
# print(w1)

# w2 = np.array([0, 4, 4])
# z2 = Xb.dot(w2)
# Y2 = sigmoid(z2)

# print("perfect solution", cross_entropy_error(T, Y2))

#z1 = Xb.dot(w1)
#Y1 = sigmoid(z1)

# print("Lin reg", cross_entropy_error(T, Y1))

# print("w", w)

# print("w1", w1)

# print("w2", w2)

# print(Xb)
#plt.scatter(range(N), Xb[:, 1],label='ass')
#plt.scatter(range(N), Xb[:, 2])

# plt.plot(Y, color='red')
# plt.scatter(range(N), Y, color='red')

#plt.plot(Y1, color='green')
# plt.scatter(range(N), Y1, color='green')


# plt.plot(Y2, color='blue')
# plt.scatter(range(N), Y2, color='blue')
# plt.show()


plt.scatter(X[:, 0], X[:, 1], c=T, s=100, alpha=0.5)

x_axis = np.linspace(-6, 6, 100)
y_axis = -x_axis
plt.plot(x_axis, y_axis)

plt.show()
