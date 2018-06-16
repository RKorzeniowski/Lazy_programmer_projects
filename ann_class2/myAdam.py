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
######################IMPORTANETE PARAMETERE ####################
    t = 1  # !!!!!!!!!!!!!!!!
###############################################################
    epochs = 20
    print_period = 10
    lr0 = 0.001
    reg = 0.01
    epsilon = 1e-8  # is it the same as 10e-8

    beta1 = 0.9  # mu = 0.9
    beta2 = 0.999  # decay = 0.999
    batch_size = 500
    number_batches = int(N // batch_size)

    W1 = np.random.randn(D, M) / 28
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K) / np.sqrt(M)
    b2 = np.zeros(K)

    tr_costs_momentum = []
    errors_batch_momentum = []
    losses_test_momentum = []

    # momentum coeficient

    mW2 = 0
    mW1 = 0
    mb2 = 0
    mb1 = 0

    vW1 = 0
    vW2 = 0
    vb1 = 0
    vb2 = 0

    mW2_hat = 0
    mW1_hat = 0
    mb2_hat = 0
    mb1_hat = 0

    vW1_hat = 0
    vW2_hat = 0
    vb1_hat = 0
    vb2_hat = 0

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

            # update momentum
            mW2 = beta1 * mW2 + (1 - beta1) * gW2
            mW1 = beta1 * mW1 + (1 - beta1) * gW1
            mb2 = beta1 * mb2 + (1 - beta1) * gb2
            mb1 = beta1 * mb1 + (1 - beta1) * gb1

            # update velocity
            vW2 = beta2 * vW2 + (1 - beta2) * gW2 * gW2
            vb2 = beta2 * vb2 + (1 - beta2) * gb2 * gb2
            vW1 = beta2 * vW1 + (1 - beta2) * gW1 * gW1
            vb1 = beta2 * vb1 + (1 - beta2) * gb1 * gb1

            # bias correction
            correction1 = (1 - beta1**t)
            mW2_hat = mW2 / correction1
            mW1_hat = mW1 / correction1
            mb2_hat = mb2 / correction1
            mb1_hat = mb1 / correction1

            correction2 = (1 - beta2**t)
            vW2_hat = vW2 / correction2
            vW1_hat = vW1 / correction2
            vb2_hat = vb2 / correction2
            vb1_hat = vb1 / correction2

            # update t !!!!!!!
            t += 1

            # update
            W2 -= lr0 * (mW2_hat / np.sqrt(vW2_hat + epsilon))
            W1 -= lr0 * (mW1_hat / np.sqrt(vW1_hat + epsilon))
            b2 -= lr0 * (mb2_hat / np.sqrt(vb2_hat + epsilon))
            b1 -= lr0 * (mb1_hat / np.sqrt(vb1_hat + epsilon))

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
