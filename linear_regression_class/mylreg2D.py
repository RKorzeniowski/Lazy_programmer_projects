import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X = []
X1 = []
X2 = []
Y = []


for line in open("data_2d.csv"):
    line = line.split(',')

# trzeba dodac x0 term do y i do x

    x1 = float(line[0])
    x2 = float(line[1])
    y = float(line[2])

    X.append([float(x1), float(x2), 1])  # x0 to caly wektor tak jak x1 i x2
    #X = ([float(x1), float(x2), 1])
    X1.append(x1)
    X2.append(x2)
    Y.append(y)

    # print " line0", line[0]
    # print " line1", line[1]


Y = np.array(Y)
#Y = Y.T
X1 = np.array(X1)
X2 = np.array(X2)
X_test = np.matrix([X1, X2])
X = np.array(X)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(X[:, 0], X[:, 1], Y)
# plt.show()


# print(len(Y))
# print(len(X1))
# print(len(X2))
print(len(X[:, 0]))
print(len(X[:, 1]))
print(len(X[:, 2]))


# print(X1)
# print("__________X2_________")
# print(X2)
# print("__________X__________")
# print(X[1, :])
# print("__________Y__________")
# print(Y)
# print("__________X__________")
# print(X_test)
# print("__________X__________")
# print(X)


#w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
print(w)

#Y_hat = X[:, 0] * w[0] + X[:, 1] * w[1] + X[:, 2] * w[2]
Y_hat = np.dot(X, w)
print(Y_hat)

# check r2

# SSres = (Y - Y_hat).dot(Y - Y_hat)
# SStot = (Y - Y.mean()).dot(Y - Y.mean())

# r2 = 1 - SSres / SStot

d1 = Y - Y_hat
d2 = Y - Y.mean()

r2 = 1 - d1.dot(d1) / d2.dot(d2)

print(r2)

# plot in 3d

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# #ax.plot(X[:, 0], X[:, 1], Y_hat)
# ax.plot_surface(X[:, 0], X[:, 1], Y_hat, alpha=0.2)
# ax.scatter(X[0], X[1], Y[0], color='green')
