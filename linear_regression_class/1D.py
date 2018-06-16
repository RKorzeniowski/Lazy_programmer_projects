import numpy as np
import matplotlib.pyplot as plt

# read csv file with numpy
# np.genformtxt(BytesIO(data),delimiter=',')
X = []
Y = []
for line in open('data_1d.csv'):
    x, y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

plt.scatter(X, Y)
plt.ylabel('Y')


data = np.genfromtxt('data_1d.csv', delimiter=',')
print(data)
x_i = data[:, 0]
# print(x_i)
y_i = data[:, 1]
# print(y_i)
N = len(x_i)
# print(N)


# solve for E
x_bar = np.sum(x_i) / N
y_bar = np.sum(y_i) / N
xy_bar = np.dot(x_i, y_i) / N
x_bar_2 = np.dot(x_i, x_i) / N
print(x_bar)
print(y_bar)
print(xy_bar)
print(x_bar_2)


ggbaby = np.sum(x_i) / N
ggbaby1 = X.sum() / N
ggbaby2 = X.sum() / N * X.mean()
ggbaby3 = x_bar * x_bar
ggbaby4 = X.mean()


print("_____________")
print(ggbaby)
print(ggbaby1)
print(ggbaby2)
print(ggbaby3)
print(ggbaby4)


denominator = (x_bar * x_bar - x_bar_2)

print("_____________a1")
a1 = (x_bar * y_bar - xy_bar) / denominator
b1 = (x_bar * xy_bar - y_bar * x_bar_2) / denominator
print(a1)
print(b1)

print("_____________a2")
denominator = X.dot(X) - X.mean() * X.sum()
a2 = (X.dot(Y) - Y.mean() * X.sum()) / denominator
b2 = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator
print(a2)
print(b2)

# prediceted Y
Ypred = a1 * x_i + b1

# plt.scatter(x_i, Ypred)
plt.plot(x_i, Ypred)
# plt.show()


# ##right but can be done on vectors
# SSres = 0
# SStot = 0
# y_mean = y_i.mean()


# for i in range(len(x_i)):
#     SSres += (y_i[i] - (x_i[i] * a1 + b1)) * (y_i[i] - (x_i[i] * a1 + b1))
#     SStot += (y_i[i] - y_mean) * (y_i[i] - y_mean)
# #    print(x_i[i])
# R = np.sqrt(1 - (SSres / SStot))


# ##R^2 = 1 perfect model, = 0  means we predicted an avrage of y, = -N means model is worse than predicting a mean
# print(SSres)
# print(SStot)
# print(R)

# vectors with all the values (same as with the loop)
#vect - vect
d1 = y_i - Ypred
# vect - scalar =np transforms it=> vect1 - vect[1 same lenght as vect1]*scalar
d2 = y_i - y_i.mean()

r2 = 1 - d1.dot(d1) / d2.dot(d2)
r = np.sqrt(r2)

print(r2)
print(r)
