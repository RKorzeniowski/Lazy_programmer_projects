import numpy as np
import matplotlib.pyplot as plt
from process import get_data
from sklearn.utils import shuffle


# input (NxD) -> W1 (MxD) -> W2 (MxK) -> output (NxK)

# dJ/dw2 = (t_k -y_k)*z_1
# T is (NxK), Z is (NxM)
# T - Y for gradient ascend
# def derivative_W2(T, Y, Z):
#     return Z.T.dot(T - Y)


def forward(X, W1, W2, b1, b2):
    Z = np.tanh(X.dot(W1) + b1)
    A = (Z.dot(W2) + b2)
    # softmax
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y, Z


def predict(P_Y_given_X):
    return np.argmax(P_Y_given_X, axis=1)


def classification_rate(Y, P):
    # n_correct = 0
    # n_total = 0
    # for i in range(len(Y)):
    #     n_total += 1
    #     if Y[i] == P[i]:
    #         n_correct += 1
    # return float(n_correct) / n_total

    return np.mean(Y == P)


def cross_entropy(T, Y):
    return -np.mean(T * np.log(Y))  # ta czesc i tak by byla rowna 0 + (1 - T) * np.log(1 - Y)


def y2indicatior(y, K):
    N = len(y)
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind


def main():

    Xtrain, Ytrain, Xtest, Ytest = get_data()
    # print(Xtest)

    Ytrain = Ytrain.astype(np.int32)
    Ytest = Ytest.astype(np.int32)
    D = Xtrain.shape[1]
    #print(N, D)
    M = 5
    K = len(set(Ytrain) | set(Ytest))  # its binary classification but with softmax

    # jeszcze zamienic Ytrain na target
#    print(Ytrain)

    # T = np.zeros((N, K))
    # for i in range(N):
    #     T[i, Ytrain[i]] = 1

    Ytrain_ind = y2indicatior(Ytrain, K)
    Ytest_ind = y2indicatior(Ytest, K)

    # initialize random weights
    W1 = np.random.randn(D, M)
    W2 = np.random.randn(M, K)
    b1 = np.zeros(M)
    b2 = np.zeros(K)

    train_costs = []
    test_costs = []
    lr = 0.001
    steps = 10000

    for epochs in range(steps):
        pYtrain, Ztrain = forward(Xtrain, W1, W2, b1, b2)
        pYtest, Ztest = forward(Xtest, W1, W2, b1, b2)
        # print(Z.shape[1])

        W2 -= lr * Ztrain.T.dot(pYtrain - Ytrain_ind)  # derivative_W2(T, pYtrain, Z)
        b2 -= lr * (pYtrain - Ytrain_ind).sum(axis=0)  # derivative_b2(T, pYtrain)
        dZ = (pYtrain - Ytrain_ind).dot(W2.T) * (1 - Ztrain * Ztrain)  # Z(1-Z) is derivitae of tanh <-#error at the hidden node
        W1 -= lr * Xtrain.T.dot(dZ)  # derivative_W1(T, pYtrain, Z, Xtrain, W2)
        b1 -= lr * dZ.sum(axis=0)  # derivative_b1(T, pYtrain, Z, W2)

        ctrain = cross_entropy(Ytrain_ind, pYtrain)  # tutaj musza byc macierze jedynek dla pojedynczej klasy
        ctest = cross_entropy(Ytest_ind, pYtest)  # tutaj musza byc macierze jedynek dla pojedynczej klasy
        train_costs.append(ctrain)
        test_costs.append(ctest)

        if epochs % 1000 == 0:
            print("cost train: ", ctrain, "cost test", ctest)

    print("final classification rate for TRANING set is:", classification_rate(Ytrain, predict(pYtrain)))
    print("final classification rate for TEST set is:", classification_rate(Ytest, predict(pYtest)))

    plt.plot(train_costs)
    plt.plot(test_costs)
    plt.show()


if __name__ == '__main__':
    main()

    # N, K = T.shape
    # M = Z.shape[1]
    # # slow way
    # ret1 = np.zeros((M, K))
    # for n in range(N):
    #     for m in range(M):
    #         for k in range(K):
    #             # cuz its gradient ascend
    #             ret1[m, k] += (T[n, k] - Y[n, k]) * Z[n, m]
    # return ret1
    # #without M loop
    # ret2 = np.zeros((M, K))
    # for n in range (N):
    #     for k in range(K):
    #         ret2[:,k] += (T[n,k] - Y[n,k]) * Z[n,:]
    # assert(np.abs(ret1 - ret2).sum() < 10e-10)

    # #also without loop K
    # ret3 = np.zeros((M,K))
    # for n in range(N):
    #     ret3 += np.outer(Z[n], T[n] - Y[n])
    #assert(np.abs(ret2 - ret3).sum() < 10e-10)

    # without any loops
    # ret4 =  Z.T.dot(T - Y) # Z.T = (MxN) (T - Y) = (NxK) so ret4 is of size MxK
    #assert(np.abs(ret4 - ret3).sum() < 10e-10)
