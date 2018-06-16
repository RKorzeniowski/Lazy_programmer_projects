import numpy as np
import matplotlib.pyplot as plt
from random import shuffle


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# def forward(X, W1, b1, W2, b2):
#     Z = sigmoid(X.dot(W1) + b1)
#     # then we count softmax
#     A = Z.dot(W2) + b2
#     expA = np.exp(A)
#     Y = expA / expA.sum(axis=1, keepdims=True)
#     return Y, Z


def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)


def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    return softmax(Z.dot(W2) + b2), Z

# classification rate
# num_corretct/num_total


def derivative_w2(Z, T, Y):
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
    # #fast way
    return Z.T.dot(T - Y)


def derivative_b2(T, Y):
    return (T - Y).sum(axis=0)


def derivative_w1(X, Z, T, Y, W2):
    N, D = X.shape
    M, K = W2.shape

    #     # slow way
    #     ret1 = np.zeros((D, M))
    #     for n in range(N):
    #         for k in range(K):
    #             for m in range(M):
    #                 for d in range(D):
    #                     ret1[d, m] += ((T[n, k] - Y[n, k]) * W2[m, k] * Z[n, m] * (1 - Z[n, m]) * X[n, d])
    # #    return ret1

    # version without d
    # ret2 = np.zeros((D, M))
    # for n in range(N):
    #     for k in range(K):
    #         for m in range(M):
    #             ret2[:, m] += ((T[n, k] - Y[n, k]) * W2[m, k] * Z[n, m] * (1 - Z[n, m]) * X[n, :])
    # #assert(np.abs(ret1 - ret2).sum() < 10e-10)
    # # return ret2

    # version without m
    # ret3 = np.zeros((D, M))
    # for n in range(N):
    #     for d in range(D):
    #         for k in range(K):
    #             ret3[d, :] += (T[n, k] - Y[n, k] * W2[:, k] * Z[n, :] * (1 - Z[n, :]) * X[n, d])

    # ret3 = np.zeros((D, M))
    # for n in range(N):
    #     for k in range(K):
    #         ret3 += np.outer(X[n, :], ((T[n, k] - Y[n, k]) * W2[:, k] * Z[n, :] * (1 - Z[n, :])))
    #    assert(np.abs(ret3 - ret2).sum() < 10e-10)
    #    return ret3

    # print(Z2.shape)

    #assert(np.abs(ret3 - ret4).sum() < 10e-10)
    #assert(np.abs(Z2 - Z1).sum() < 10e-10)
    # return ret4

    # derive how to get from slow to fast here (same way as in derivative_W2)
    dZ = (T - Y).dot(W2.T) * Z * (1 - Z)
    return X.T.dot(dZ)

# fast way ((derive how we did it HOMEWORD))


def derivative_b1(T, Y, W2, Z):
    return ((T - Y).dot(W2.T) * Z * (1 - Z)).sum(axis=0)

# def derivative_b1(T, Y, W2, Z):
#     return ((T - Y).dot(W2.T) * Z * (1 - Z)).sum(axis=0)


def onehot(Y, N, K):
    T = np.zeros((N, K))
    for i in range(N):
        T[i, Y[i]] = 1


def y2indicator(Y, K):
    N = len(Y)
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, Y[i]] = 1
    return ind


def cost(T, Y):
    tot = T * np.log(Y)
    return tot.sum()


def classificaiton_rate(Y, P):
    n_correct = 0
    n_total = 0
    # print(P.shape)
    # print(Y.shape)
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct) / n_total


def main():

    Nclass = 500
    D = 2  # dimentinality of input
    M = 3  # hidden layer size
    K = 3  # number of classes

    # create classes , the data
    X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
    X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
    X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
    X = np.vstack([X1, X2, X3])

    # create lables
    Y = np.array([0] * Nclass + [1] * Nclass + [2] * Nclass)

    N = len(Y)

    # turn targets into a indicatior
    # its like onehotencodeing for the targers

    # plot the data to see how it looks like
    #plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
    # plt.show()

    # randomy initializie weights
    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    #X = np.random.shuffle(X)
    #Y = np.random.shuffle(Y)

    K_Folds = 4
    sz = int(N / K_Folds)
    print(sz)

    learning_rate = 10e-7
    costs = []
    class_rates = []
    costs_test = []
    class_rates_test = []
    steps = 30000
    for k in range(K_Folds):

        W1 = np.random.randn(D, M)
        b1 = np.random.randn(M)
        W2 = np.random.randn(M, K)
        b2 = np.random.randn(K)

        print(k)
        xtr = np.concatenate([X[:k * sz:, :], X[(k * sz + sz):, :]])
        ytr = np.concatenate([Y[:k * sz], Y[(k * sz + sz):]])
        xte = X[k * sz:(k * sz + sz), :]
        yte = Y[k * sz:(k * sz + sz)]

        print(xtr.shape)
        print(ytr.shape)
        print(xte.shape)
        print(yte.shape)

        ttr = y2indicator(ytr, K)
        tte = y2indicator(yte, K)
# sometimes classification rate goes to 0 when cost is quete ok and it probably means that there is something wrong with classification

        # print cost and classification rate evert 100 steps
        for epoch in range(steps):
            Y_hat, Z = forward(xtr, W1, b1, W2, b2)
            Y_hat_test, Z_test = forward(xte, W1, b1, W2, b2)
            if epoch % 100 == 0:
                # if epoch == (10000 - 2):
                c = cost(ttr, Y_hat)
                c_test = cost(tte, Y_hat_test)
                P = np.argmax(Y_hat, axis=1)
                P_test = np.argmax(Y_hat_test, axis=1)
                # print(P.shape)
                # print(Y_hat.shape)
                r = classificaiton_rate(ytr, P)
                r_test = classificaiton_rate(yte, P_test)  # Y_hat ma teraz rozmiar zmiejszonej a Y nie jest zmeijszony
                print("cost:", c, "classification rate:", r)
                class_rates.append(r)
                costs.append(c)
                print("cost test:", c_test, "classification rate:", r_test)
                class_rates_test.append(r_test)
                costs_test.append(c_test)
            # gradient ascend (backpropagation)
            W2 += learning_rate * derivative_w2(Z, ttr, Y_hat)
            b2 += learning_rate * derivative_b2(ttr, Y_hat)
            W1 += learning_rate * derivative_w1(xtr, Z, ttr, Y_hat, W2)
            b1 += learning_rate * derivative_b1(ttr, Y_hat, W2, Z)

        plt.plot(costs, label='train')
        plt.plot(costs_test, label='test')
        plt.legend()
        plt.show()
        plt.plot(class_rates, label='train')
        plt.plot(class_rates_test, label='test')
        plt.legend()
        plt.show()
# plot and low == value or mean and == calue = to steps-1
    # print("################################")
    # print("cost mean", np.mean(costs), "std", np.std(costs), "cost", costs)
    # print("################################")
    # print("class_rates mean", np.mean(class_rates), "std", np.std(class_rates), "class_rates", class_rates)
    # print("################################")
    # print("cost mean test", np.mean(costs_test), "std", np.std(costs_test), "cost", costs_test)
    # print("################################")
    # print("class_rates mean test", np.mean(class_rates_test), "std", np.std(class_rates_test), "class_rates", class_rates_test)
    # print("################################")

    # scores.append(np.argmin(costs))  # ja chce najwiekszy
    #score = cost(tte, Y_hat_test)
    # scores.append(score)
    #print(" traning ", np.mean(scores), np.std(scores))

    #print("scores", scores)
    #print(" traning ", np.mean(scores), np.std(scores))

    # plt.plot(costs)
    # plt.show()


if __name__ == '__main__':
    main()
