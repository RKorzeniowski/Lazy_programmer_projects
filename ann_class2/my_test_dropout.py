import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from util import get_normalized_data, y2indicator


def error_rate(p, t):
    return np.mean(p != t)

# uzycie dropoutu tutaj moze nie miec sensu


def main():

    # load data
    X, Y = get_normalized_data()

    # split to traning and test set
    # pick all except last 1000 elements from X & Y
    Xtrain = X[:-1000, ]
    Ytrain = Y[:-1000]
    # pick only last 1000 elements from X & Y
    Xtest = X[-1000:, ]
    Ytest = Y[-1000:]
    # Y as indicatior matrix
    #[ 3      [ 0 0 1 0
    #  2  -->   0 1 0 0
    #  1        1 0 0 0
    #  4]       0 0 0 1]
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    N, D = Xtrain.shape

    M1 = 600
    M2 = 300
    M3 = 150
    M4 = 80
    K = 10

    # paramters of traning
    steps = 15
    print_period = 10
    #lr = 0.00004

    beta = 0.005
    batch_sz = 500

    lr = 0.001  # / batch_sz

    # initialize weights and biases and normalize them to sum to 1
    W1_init = np.random.randn(D, M1) / 28
    b1_init = np.zeros(M1)
    W2_init = np.random.randn(M1, M2) / np.sqrt(M1)
    b2_init = np.zeros(M2)
    W3_init = np.random.randn(M2, M3) / np.sqrt(M2)
    b3_init = np.zeros(M3)
    W4_init = np.random.randn(M3, M4) / np.sqrt(M3)
    b4_init = np.zeros(M4)
    W5_init = np.random.randn(M4, K) / np.sqrt(M4)
    b5_init = np.zeros(K)

    #iniatialize in tf
    X = tf.placeholder(tf.float32, shape=(None, D), name='X')  # value is not needed here only what kind of var is it
    T = tf.placeholder(tf.float32, shape=(None, K), name='T')  # value is not needed here only what kind of var is it
    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))
    W4 = tf.Variable(W4_init.astype(np.float32))
    b4 = tf.Variable(b4_init.astype(np.float32))
    W5 = tf.Variable(W5_init.astype(np.float32))
    b5 = tf.Variable(b5_init.astype(np.float32))

    logits_1 = tf.matmul(X, W1) + b1
    Z1 = tf.nn.relu(logits_1)  # output of hidden layer 1
    # dropout on hidden layer 1
    keep_prob = tf.placeholder("float")
    Z1_dropout = tf.nn.dropout(Z1, keep_prob)

    Z2 = tf.nn.relu(tf.matmul(Z1_dropout, W2) + b2)  # output of hidden layer 2
    Z2_dropout = tf.nn.dropout(Z2, keep_prob)  # dropout on hidden layer 2
    Z3 = tf.nn.relu(tf.matmul(Z2_dropout, W3) + b3)  # output of hidden layer 3
    Z3_dropout = tf.nn.dropout(Z3, keep_prob)  # dropout on hidden layer 3
    Z4 = tf.nn.relu(tf.matmul(Z3_dropout, W4) + b4)  # output of hidden layer 4
    Z4_dropout = tf.nn.dropout(Z4, keep_prob)  # dropout on hidden layer 4
    # its not really Y. Its matrix multiplication of Z2 and W3 without doing the softmax (cuz its included in cost calcucation)
    Yish = tf.matmul(Z4_dropout, W5) + b5  # its a logit: Unscaled log probabilities.

    # calculate cost with cross entrpy for softmax
    # loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Yish, labels=T))  # this does not divide by the number of elements in the batch is lr has to be adjusted (super small)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Yish, labels=T))  # this one does so its ok with normal lr

    # reg added to the cost function
    regularizer = tf.nn.l2_loss(W5) + tf.nn.l2_loss(W4) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W1)
    cost = tf.reduce_mean(loss + beta * regularizer)

    # define opimizer
    train_op = tf.train.RMSPropOptimizer(lr, decay=0.99, momentum=0.9).minimize(cost)

    predict_op = tf.argmax(Yish, 1)

    n_batches = N // batch_sz
    costs = []
    init = tf.global_variables_initializer()
    # traning loop
    with tf.Session() as session:
        session.run(init)

        for i in range(steps):
            for j in range(n_batches):
                Xbatch = Xtrain[j * batch_sz:(batch_sz + j * batch_sz), ]
                Ybatch = Ytrain_ind[j * batch_sz:(batch_sz + j * batch_sz), ]

                session.run(train_op, feed_dict={X: Xbatch, T: Ybatch, keep_prob: 0.5})  # tu jest ok ze keep_prob = 0.5
                if j % print_period == 0:
                    test_cost = session.run(cost, feed_dict={X: Xtest, T: Ytest_ind, keep_prob: 0.5})  # ale czy tutaj i nizej nie powinienem uzyc jakiejs macierzy ktora wybiera najlepsze albo cos
                    prediction = session.run(predict_op, feed_dict={X: Xtest, keep_prob: 0.5})
                    err = error_rate(prediction, Ytest)
                    print(i, j, test_cost, err)
                    costs.append(test_cost)

    plt.plot(costs)
    plt.show()


if __name__ == '__main__':
    main()
