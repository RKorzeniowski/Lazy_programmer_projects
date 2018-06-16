import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_excel("mlr02.xls")
X = df.as_matrix()


# plt.scatter(X[:, 1], X[:, 0])
# plt.show()


# plt.scatter(X[:, 2], X[:, 0])
# plt.show()

def get_r2(X, Y):

    w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
    print(w)
    Y_hat = np.dot(X, w)
    d1 = Y - Y_hat
    d2 = Y - Y.mean()
    r2 = 1 - d1.dot(d1) / d2.dot(d2)
    return r2


df['ones'] = 1

print(df)

Y = df['X1']
X = df[['X2', 'X3', 'ones']]


X = np.array(X)
Y = np.array(Y)
X2 = df[['X2', 'ones']]
X2 = np.array(X2)
X3 = df[['X3', 'ones']]
X3 = np.array(X3)


noise = np.random.normal(0, 1, len(Y))
df['noise'] = noise
print("noise", noise)
Xnoise = df[['X2', 'X3', 'ones', 'noise']]
Xnoise = np.array(Xnoise)

r21 = get_r2(X2, Y)
print("X2 r2", r21)
r22 = get_r2(X3, Y)
print("X3 r2", r22)
r2n = get_r2(Xnoise, Y) #adding noise collumn improves r2
print(r2n)
r2 = get_r2(X, Y)
print(r2)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:, 0], X[:, 1], Y_hat)
# plt.show()
