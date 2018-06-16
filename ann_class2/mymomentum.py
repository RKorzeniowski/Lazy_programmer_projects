import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from util import get_normalized_data, get_transformed_data, forward, error_rate, cost, gradW, gradb, y2indicator
from mlp import forward, derivative_w2, derivative_w1, derivative_b1, derivative_b2


def main():

    # compare 3:
    # 1. batch SGD
    # 2. batch SGD with momentum
    # 3. batch SGD with Nesterov momentum
    # all with L2 regularization

    print_period = 10

    X, Y = get_normalized_data()
    lr = 0.00004

    reg = 0.01

    Xtrain = X[:-1000, ]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:, ]
    Ytest = Y[-1000:]

    N, D = Xtrain.shape
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    M = 300
    K = 10

    W1 = np.random.randn(D, M) / 28
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K) / np.sqrt(M)
    b2 = np.zeros(K)

    W1_0 = W1.copy()
    b1_0 = b1.copy()
    W2_0 = W2.copy()
    b2_0 = b2.copy()

    # regular batch gradient descend

    epochs = 30

    tr_costs = []
    errors_batch = []
    losses_test = []
    batch_size = 500
    number_batches = int(N // batch_size)

    #max_iter = 30
    # 1.
    for epoch in range(epochs):
        for j in range(number_batches):
            xtr = Xtrain[j * batch_size:(j * batch_size + batch_size), :]
            ytr = Ytrain_ind[j * batch_size:(j * batch_size + batch_size), :]
            ytr_pred, z_tr = forward(xtr, W1, b1, W2, b2)

            W2 -= lr * (derivative_w2(z_tr, ytr, ytr_pred) + reg * W2)
            b2 -= lr * (derivative_b2(ytr, ytr_pred) + reg * b2)
            W1 -= lr * (derivative_w1(xtr, z_tr, ytr, ytr_pred, W2) + reg * W1)
            b1 -= lr * (derivative_b1(z_tr, ytr, ytr_pred, W2) + reg * b1)

            if j % print_period == 0:
                yte_pred, _ = forward(Xtest, W1, b1, W2, b2)
                l = cost(yte_pred, Ytest_ind)
                losses_test.append(l)
                print("test set Cost at iteration epoch=%d, j=%d: %.6f" % (epoch, j, l))

                e = error_rate(yte_pred, Ytest)
                errors_batch.append(e)
                print("Error rate:", e)

                ctr = cost(ytr_pred, ytr)
                print("traning set cost", ctr)
                tr_costs.append(ctr)

    pY, _ = forward(Xtest, W1, b1, W2, b2)

    #plt.plot(tr_costs, label='tr_costs')
    plt.plot(losses_test, label='losses_test')
    #plt.plot(errors_batch, label='errors_batch')
#    plt.show()
#    print("tr_costs", tr_costs)
    print("Final error rate:", error_rate(pY, Ytest))

    # 2.
    W1 = W1_0.copy()
    b1 = b1_0.copy()
    W2 = W2_0.copy()
    b2 = b2_0.copy()

    # regular batch gradient descend

    tr_costs_momentum = []
    errors_batch_momentum = []
    losses_test_momentum = []

    # momentum coeficient
    mu = 0.9

    dW1 = 0
    dW2 = 0
    db1 = 0
    db2 = 0

    for epoch in range(epochs):
        for j in range(number_batches):
            xtr = Xtrain[j * batch_size:(j * batch_size + batch_size), :]
            ytr = Ytrain_ind[j * batch_size:(j * batch_size + batch_size), :]
            ytr_pred, z_tr = forward(xtr, W1, b1, W2, b2)

            # gradients
            gW2 = derivative_w2(z_tr, ytr, ytr_pred) + reg * W2
            gb2 = derivative_b2(ytr, ytr_pred) + reg * b2
            gW1 = derivative_w1(xtr, z_tr, ytr, ytr_pred, W2) + reg * W1
            gb1 = derivative_b1(z_tr, ytr, ytr_pred, W2) + reg * b1

            # update velocity
            dW2 = mu * dW2 - lr * gW2
            db2 = mu * db2 - lr * gb2
            dW1 = mu * dW1 - lr * gW1
            db1 = mu * db1 - lr * gb1

            # update
            W2 += dW2
            W1 += dW1
            b2 += db2
            b1 += db1
            if j % print_period == 0:
                yte_pred, _ = forward(Xtest, W1, b1, W2, b2)
                l = cost(yte_pred, Ytest_ind)
                losses_test_momentum.append(l)
                print("test set Cost at iteration epoch=%d, j=%d: %.6f" % (epoch, j, l))

                e = error_rate(yte_pred, Ytest)
                errors_batch_momentum.append(e)
                print("Error rate:", e)

                ctr = cost(ytr_pred, ytr)
                print("traning set cost", ctr)
                tr_costs_momentum.append(ctr)

    pY, _ = forward(Xtest, W1, b1, W2, b2)

    #plt.plot(tr_costs_momentum, label='tr_costs momentum')
    plt.plot(losses_test_momentum, label='losses_test momentum')
    #plt.plot(errors_batch, label='errors_batch')
    # plt.show()
#    print("tr_costs", errors_batch_momentum)
    print("Final error rate:", error_rate(pY, Ytest))


# 3.

    W1 = W1_0.copy()
    b1 = b1_0.copy()
    W2 = W2_0.copy()
    b2 = b2_0.copy()

    # regular batch gradient descend

    tr_costs_nesterov = []
    errors_batch_nesterov = []
    losses_test_nesterov = []

    # momentum coeficient
    mu = 0.9

    vW1 = 0
    vW2 = 0
    vb1 = 0
    vb2 = 0

    for epoch in range(epochs):
        for j in range(number_batches):
            xtr = Xtrain[j * batch_size:(j * batch_size + batch_size), :]
            ytr = Ytrain_ind[j * batch_size:(j * batch_size + batch_size), :]
            ytr_pred, z_tr = forward(xtr, W1, b1, W2, b2)

            # gradients
            gW2 = derivative_w2(z_tr, ytr, ytr_pred) + reg * W2
            gb2 = derivative_b2(ytr, ytr_pred) + reg * b2
            gW1 = derivative_w1(xtr, z_tr, ytr, ytr_pred, W2) + reg * W1
            gb1 = derivative_b1(z_tr, ytr, ytr_pred, W2) + reg * b1

            # update velocity
            vW2 = mu * vW2 - lr * gW2
            vb2 = mu * vb2 - lr * gb2
            vW1 = mu * vW1 - lr * gW1
            vb1 = mu * vb1 - lr * gb1

            # update
            W2 += mu * vW2 - lr * gW2
            W1 += mu * vW1 - lr * gW1
            b2 += mu * vb2 - lr * gb2
            b1 += mu * vb1 - lr * gb1

            if j % print_period == 0:
                yte_pred, _ = forward(Xtest, W1, b1, W2, b2)
                l = cost(yte_pred, Ytest_ind)
                losses_test_nesterov.append(l)
                print("test set Cost at iteration epoch=%d, j=%d: %.6f" % (epoch, j, l))

                e = error_rate(yte_pred, Ytest)
                errors_batch_nesterov.append(e)
                print("Error rate:", e)

                ctr = cost(ytr_pred, ytr)
                print("traning set cost", ctr)
                tr_costs_nesterov.append(ctr)

    pY, _ = forward(Xtest, W1, b1, W2, b2)

    #plt.plot(tr_costs_nesterov, label='tr_costs_nesterov')
    plt.plot(losses_test_nesterov, label='losses_test_nesterov')
    #plt.plot(errors_batch_nesterov, label='errors_batch')
    plt.legend()
    plt.show()
#    print("tr_costs_nesterov", errors_batch_momentum)
    print("Final error rate nesterov:", error_rate(pY, Ytest))


if __name__ == '__main__':
    main()
