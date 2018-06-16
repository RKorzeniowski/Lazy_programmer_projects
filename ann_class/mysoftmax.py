import numpy as np

# single prediction with an array
a = np.random.randn(5)  # output of a ANN| 5 different numbers that reprezent activation of 5 different nodes
print(a)

max_value = np.exp(a)
print(max_value)

softmax_values = []

for i in range(len(a)):
    softmax_values.append(np.exp(a[i]) / max_value.sum())


print(softmax_values)
print(np.asarray(softmax_values).sum())
print(np.argmax(softmax_values))

pred = np.zeros(len(a))

pred[np.argmax(softmax_values)] = 1
print(pred)


# matrix of elements N=10 samples K=5 so 5 classes
N = 10
K = 5
A = np.random.randn(N, K)
print(A)

expA = np.exp(A)

# divie every cell by the sum of the cells on the axis=1(row)

pred = expA / expA.sum(axis=1, keepdims=True)
print(pred)

print(pred.sum(axis=1))
print(expA.sum(axis=1, keepdims=True))

# softmax_values = np.zeros([N, K])

# for j in range(K):
#     diverderino = np.exp(A[:, j])
#     for i in range(N):
#         softmax_values = np.exp(A[i, j]) / diverderino

# pred = np.zeros([N, K])

# print(np.argmax(softmax_values[:, 1]))

# for i in range(N):
#     pred[np.argmax(softmax_values), i] = 1  # pytanie brzmi czy to zwroci array i czy ten array zadziala jako wskaznik


# print(pred)


onehotencoder = np.zeros([N, K])
idx = np.argmax(pred, axis=1)
print(idx)

for i in range(N):
    onehotencoder[i, idx[i]] = 1

print(onehotencoder)
