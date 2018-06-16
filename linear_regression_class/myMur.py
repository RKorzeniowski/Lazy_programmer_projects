
# import pandas as pd

# df = pd.read_csv("moore.csv")
# print(df)


# for line in open("moore.csv"):
#     z, x, y = line.split(',')
#     X.append(float(x))
#     Y.append(float(y))


# import csv
# from collections import defaultdict

# columns = defaultdict(list)  # each value in each column is appended to a list

# with open("moore.csv") as f:
#     reader = csv.DictReader(f)  # read rows into a dictionary format
#     for row in reader:  # read a row as {column1: value1, column2: value2,...}
#         for (k, v) in row.items():  # go over each column name and value
#             columns[k].append(v)

# print columns


import re
import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []

non_decimal = re.compile(r'[^\d]+')  # remove all non decimal charachters from the input

for line in open("moore.csv"):
    r = line.split('\t')

    x = int(non_decimal.sub('', r[2].split('[')[0]))  # year
    y = int(non_decimal.sub('', r[1].split('[')[0]))  # transistors number

    X.append(float(x))
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)
#plt.scatter(X, Y)
# plt.show()
Y = np.log(Y)
#plt.scatter(X, Y)
# plt.show()


# print X, Y
X_mean = X.mean()
Y_mean = Y.mean()
X_sum = X.sum()

denominator = X_mean * X_sum - np.dot(X, X)
a = (Y_mean * X_sum - X.dot(Y)) / denominator
b = (X_mean * X.dot(Y) - Y_mean * X.dot(X)) / denominator
print a, b
# predicted line
Y_hat = a * X + b

plt.scatter(X, Y)
plt.plot(X, Y_hat)
# plt.show()

# how good is the model

d1 = Y - Y_hat
d2 = Y - Y.mean()

r2 = 1 - d1.dot(d1) / d2.dot(d2)

print r2

# log(tc) = a*year + b
# tc = exp(b) * exp(a * year)
# 2*tc = 2*exp(b) * exp(a * year) = exp(ln(2)) * exp(b) * exp(a * year)
#      = exp(ln(2) + a * year) * exp(b)
# exp(a * year2) * exp(b) = exp(ln(2) + a * year) * exp(b)
# a*year2 = a*year1 + ln(2)
# year2 - year1 = ln(2)/a
print("time to double number of transistors %i", np.log(2) / a)

# log(2*tc) = a*(year2 - year1) + b
