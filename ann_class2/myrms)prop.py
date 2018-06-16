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

    X, Y = get_normalized_data()

    Xtrain = X[:-1000, ]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:, ]
    Ytest = Y[-1000:]

    N, D = Xtrain.shape
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    M = 300
    K = 10

    max_iter = 20
    epochs = 20
    print_period = 10
    lr0 = 0.0004
    reg = 0.01
    epsilon = 10e-10
    decay = 0.999
    batch_size = 500
    number_batches = int(N // batch_size)

    W1 = np.random.randn(D, M) / 28
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K) / np.sqrt(M)
    b2 = np.zeros(K)

    W1_0 = W1.copy()
    b1_0 = b1.copy()
    W2_0 = W2.copy()
    b2_0 = b2.copy()

    cache_W2 = 1
    cache_W1 = 1
    cache_b2 = 1
    cache_b1 = 1

    tr_costs = []
    errors_batch = []
    losses_test = []

# 1. Just grad & RMSprop

   # 1.
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

            # # AdaGrad
            # cache_W2 += derivative_w2(z_tr, ytr, ytr_pred) * derivative_w2(z_tr, ytr, ytr_pred)
            # cache_W1 += derivative_w1(xtr, z_tr, ytr, ytr_pred, W2) * derivative_w1(xtr, z_tr, ytr, ytr_pred, W2)
            # cache_b2 += derivative_b2(ytr, ytr_pred) * derivative_b2(ytr, ytr_pred)
            # cache_b1 += derivative_b1(z_tr, ytr, ytr_pred, W2) * derivative_b1(z_tr, ytr, ytr_pred, W2)

            # RMSProp
            cache_W2 += decay * cache_W2 + (1 - decay) * gW2 * gW2
            cache_W1 += decay * cache_W1 + (1 - decay) * gW1 * gW1
            cache_b2 += decay * cache_b2 + (1 - decay) * gb2 * gb2
            cache_b1 += decay * cache_b1 + (1 - decay) * gb1 * gb1

            W2 -= lr0 * (gW2 // (cache_W2 + epsilon) + reg * W2)
            b2 -= lr0 * (gb2 // (cache_b2 + epsilon) + reg * b2)
            W1 -= lr0 * (gW1 // (cache_W1 + epsilon) + reg * W1)
            b1 -= lr0 * (gb1 // (cache_b1 + epsilon) + reg * b1)

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
    plt.plot(losses_test, label='losses_test RMS')
    #plt.plot(errors_batch, label='errors_batch')
#    plt.show()
#    print("tr_costs", tr_costs)
    print("Final error rate:", error_rate(pY, Ytest))


# 2. batch grad with momentum & RMSprop
#
#
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
    mu = 0.8

    cache_W2 = 1
    cache_W1 = 1
    cache_b2 = 1
    cache_b1 = 1

    dW1 = 0
    dW2 = 0
    db1 = 0
    db2 = 0

    cW1 = 0
    cW2 = 0
    cb1 = 0
    cb2 = 0

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

            # potencjalnie pojebalem momentum i velocity

            # RMSProp
            cache_W2 = decay * cache_W2 + (1 - decay) * gW2 * gW2
            cache_W1 = decay * cache_W1 + (1 - decay) * gW1 * gW1
            cache_b2 = decay * cache_b2 + (1 - decay) * gb2 * gb2
            cache_b1 = decay * cache_b1 + (1 - decay) * gb1 * gb1

            cW2 = (gW2 // (cache_W2) + epsilon)
            cb2 = (gb2 // (cache_b2) + epsilon)
            cW1 = (gW1 // (cache_W1) + epsilon)
            cb1 = (gb1 // (cache_b1) + epsilon)

            # update velocity
            dW2 = mu * dW2 + (1 - mu) * lr0 * cW2
            db2 = mu * db2 + (1 - mu) * lr0 * cb2
            dW1 = mu * dW1 + (1 - mu) * lr0 * cW1
            db1 = mu * db1 + (1 - mu) * lr0 * cb1

            # update
            W2 -= dW2
            W1 -= dW1
            b2 -= db2
            b1 -= db1

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
    plt.plot(losses_test_momentum, label='losses_test momentum RMS')
    #plt.plot(errors_batch, label='errors_batch')
    # plt.show()
#    print("tr_costs", errors_batch_momentum)
    print("Final error rate:", error_rate(pY, Ytest))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
