from util import getKaggleMNIST
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle


class Autoencoder:

    def __init__(self, D, M):
        # create placeholder for batch
        self.X = tf.placeholder(tf.float32, shape=(None, D))
        # create weights and baiases for layers input -> hidden -> output
        # input -> hidden
        self.W = tf.Variable(tf.random_normal(shape=(D, M)) * np.sqrt(2.0 / M))
        self.b = tf.Variable(np.zeros(M).astype(np.float32))
        # hidden -> output
        self.Wo = tf.Variable(tf.random_normal(shape=(M, D)) * np.sqrt(2.0 / D))
        self.bo = tf.Variable(np.zeros(D).astype(np.float32))

        # create forward function and get logits
        self.Z = tf.nn.relu(tf.matmul(self.X, self.W) + self.b)
        logits = tf.matmul(self.Z, self.Wo) + self.bo
        self.output = tf.nn.sigmoid(logits)
        # create cost function from logits | tf.reduce_sum (reduce_sum = reduce_mean * batch_sz)
        self.cost = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.X,
                logits=logits
            )
        )
        # create training optimizer from cost
        self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.cost)
        #self.train_op = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(self.cost)

        # initialize parameters for tf session and tf.session
        # # set up session and variables for later
        self.init = tf.global_variables_initializer()
        # difference
        #sess = tf.Session.run(init)
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)

    def fit(self, X, batch_sz=100, epochs=10, plot_fig=False):
        N, _ = X.shape
        # print("N: ", N)
        # print("len(X)", len(X))
        n_batches = N // batch_sz
        print("n_batches:", n_batches)
        costs = []
        for i in range(epochs):
            print("epoch: ", epochs)
            # np.random.shuffle(X)
            for j in range(n_batches):
                batch = X[j * batch_sz:(j * batch_sz + batch_sz)]
                _, c = self.sess.run((self.train_op, self.cost), feed_dict={self.X: batch})
                c /= batch_sz  # wut?
                costs.append(c)
                if j % 10 == 0:
                    print("iter: %d, cost: %.3f" % (j, c))
        if plot_fig:
            plt.plot(costs)
            plt.show()

    def predict(self, X):
        return self.sess.run(self.output, feed_dict={self.X: X})


def singleAE():
    X, Y, _, _ = getKaggleMNIST()

    N, D = X.shape
    #print("D ", D)
    AE = Autoencoder(D, 20)
    AE.fit(X, plot_fig=True)

    done = False
    while not done:
        i = np.random.choice(len(X))
        x = X[i]
        # print(x)
        im = AE.predict([x]).reshape(28, 28)
        plt.subplot(1, 2, 1)
        plt.imshow(x.reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.subplot(1, 2, 2)
        plt.imshow(im, cmap='gray')
        plt.title("Reconstruction")
        plt.show()

        ans = input("Generate another?")
        if ans and ans[0] in ('n' or 'N'):
            done = True


if __name__ == '__main__':
    singleAE()
