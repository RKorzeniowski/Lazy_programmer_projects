import numpy as np

A = np.matrix('1 4; 9 16')
B = np.matrix('1 2; 3 4;5,6')

#C = A * B
# print(C)

# D = A.T.dot(B)
# print(D)


# E = A.T.dot(B)
# print(E)

A = np.asarray(A)
B = np.asarray(B)

# C = B * A  # so a[i] * b[i]
# print(C)

D = B.dot(A)  # normal matrix multiplication
print(D)
