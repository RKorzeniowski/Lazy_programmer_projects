import numpy as np
import theano.tensor as T
import theano
import matplotlib.pyplot as plt

from util import get_normalized_data, y2indicator


def error_rate(p, t):
    return np.mean(p != t)


def relu(a):
    return a * (a > 0)


def main():
    X, Y = get_normalized_data()

    Xtrain = X[:-1000, ]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:, ]
    Ytest = Y[-1000:]

    N, D = Xtrain.shape
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    epochs = 20
    print_period = 10
    lr = 0.0004
    reg = 0.01
    batch_sz = 500
    number_batches = int(N // batch_sz)

    M = 300
    K = 10
    W1_init = np.random.randn(D, M) / 28
    b1_init = np.zeros(M)
    W2_init = np.random.randn(M, K) / np.sqrt(M)
    b2_init = np.zeros(K)

    thX = T.matrix('X')
    thT = T.matrix('T')
    W1 = theano.shared(W1_init, 'W1')
    W2 = theano.shared(W2_init, 'W2')
    b1 = theano.shared(b1_init, 'b1')
    b2 = theano.shared(b2_init, 'b2')

    thZ = relu(thX.dot(W1) + b1)
    thY = T.nnet.softmax(thZ.dot(W2) + b2)

    cost = -(thT * T.log(thY)).sum() + reg * ((W1 * W1).sum() + (b1 * b1).sum() + (W2 * W2).sum() + (b2 * b2).sum())
    prediction = T.argmax(thY, axis=1)

    update_W1 = W1 - lr * T.grad(cost, W1)
    update_W2 = W2 - lr * T.grad(cost, W2)
    update_b1 = b1 - lr * T.grad(cost, b1)
    update_b2 = b2 - lr * T.grad(cost, b2)

    train = theano.function(
        inputs=[thX, thT],
        updates=[(W1, update_W1), (b1, update_b1), (W2, update_W2), (b2, update_b2)],
    )

    get_prediction = theano.function(
        inputs=[thX, thT],
        outputs=[cost, prediction],
    )

    LL = []
    for epoch in range(epochs):
        for j in range(number_batches):
            Xbatch = Xtrain[j * batch_sz:(j * batch_sz + batch_sz), ]
            Ybatch = Ytrain_ind[j * batch_sz:(j * batch_sz + batch_sz), ]

            train(Xbatch, Ybatch)
            if j % print_period == 0:
                cost_val, prediction_val = get_prediction(Xtest, Ytest_ind)
                err = error_rate(prediction_val, Ytest)  # why use error when
                print("test set Cost at iteration epoch=%d, j=%d: %.6f" % (epoch, j, cost_val))
                LL.append(cost_val)

    plt.plot(LL)


if __name__ == '__main__':
    main()
