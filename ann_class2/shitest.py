import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from util import get_normalized_data
from sklearn.utils import shuffle


class HiddenLayer(object):
    """docstring for HiddenLayer
    M1 input size of a layer
    M2 output size of a layer
    """

    def __init__(self, M1, M2):
        self.M1 = M1
        self.M2 = M2
        W_init = np.random.randn(M1, M2) * np.sqrt(2.0 / M1)
        b_init = np.zeros(M2)
        self.W = tf.Variable(W_init.astype(np.float32))
        self.b = tf.Variable(b_init.astype(np.float32))
        self.params = [self.W, self.b]

    def forward(self, X):
        return tf.nn.relu(tf.matmul(X, self.W) + self.b)


class ANN(object):
    """docstring for ANN
    hidden_layer_sizes is a vector e.g. [20,50,100]
    p_keep is a probability of keeping neuron

    """

    def __init__(self, hidden_layer_sizes, p_keep):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rates = p_keep

    def fit(self, X, Y, lr=0.0008, beta=0.01, decay=0.89, mu=0.9, epochs=15, batch_sz=500, print_period=20, split=True):

        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Y = Y.astype(np.int64)
        if split:
            Xvalid, Yvalid = X[-1000:], Y[-1000:]
            X, Y = X[:-1000], Y[:-1000]
        else:
            Xvalid, Yvalid = X, Y

        # uzyc classy hidden do budowania hidden layerow
        N, D = X.shape
        K = len(set(Y))
        self.hidden_layers = []
        M1 = D
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2)
            self.hidden_layers.append(h)
            M1 = M2
        W = np.random.randn(M1, K) * np.sqrt(2.0 / M1)
        b = np.zeros(K)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))

        # collect all params for later use
        self.params = [self.W, self.b]
        for h in self.hidden_layers:
            self.params += h.params

        # setup functions
        inputs = tf.placeholder(tf.float32, shape=(None, D), name='inputs')
        labels = tf.placeholder(tf.int64, shape=(None,), name='labels')
        logits = self.forward(inputs)

        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

        # different traning optimizers
        train_op = tf.train.RMSPropOptimizer(lr, decay=decay, momentum=mu).minimize(cost)
        #train_op = tf.train.MomentumOptimizer(lr, momentum=mu).minimize(cost)
        #train_op = tf.train.AdamOptimizer(lr).minimize(cost)
        prediction = self.predict(inputs)

        test_logits = self.forward_test(inputs)
        test_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=test_logits, labels=labels))

        n_batches = N // batch_sz
        costs = []
        init = tf.global_variables_initializer()
        # traning loop
        with tf.Session() as session:
            session.run(init)

            for i in range(epochs):
                print("epoch:", i, "n_batches:", n_batches)
                X, Y = shuffle(X, Y)
                for j in range(n_batches):
                    Xbatch = X[j * batch_sz:(batch_sz + j * batch_sz)]
                    Ybatch = Y[j * batch_sz:(batch_sz + j * batch_sz)]

                    session.run(train_op, feed_dict={inputs: Xbatch, labels: Ybatch})  # TUTAJ JEST INACZEJ {inputs: Xbatch, labels: Ybatch}
                    if j % print_period == 0:
                        c = session.run(test_cost, feed_dict={inputs: Xvalid, labels: Yvalid})
                        p = session.run(prediction, feed_dict={inputs: Xvalid})
                        e = error_rate(Yvalid, p)
                        print("i:", i, "j:", j, "nb:", n_batches, "cost:", c, "error rate:", e)
                        costs.append(c)

        plt.plot(costs)
        plt.show()

    def forward(self, X):
        # tf.nn.dropout scales inputs by 1/p_keep
        # therefore, during test time, we don't have to scale anything
        Z = X
        Z = tf.nn.dropout(Z, self.dropout_rates[0])
        for h, p in zip(self.hidden_layers, self.dropout_rates[1:]):
            Z = h.forward(Z)
            Z = tf.nn.dropout(Z, p)
        return tf.matmul(Z, self.W) + self.b

    def forward_test(self, X):
        Z = X
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return tf.matmul(Z, self.W) + self.b

    def predict(self, X):
        pY = self.forward_test(X)
        return tf.argmax(pY, 1)


def error_rate(p, t):
    return np.mean(p != t)


if __name__ == '__main__':
    # load data
    X, Y = get_normalized_data()

    # initialize model
    ann = ANN([100, 50], [0.8, 0.5, 0.5])

    # fit
    ann.fit(X, Y)
