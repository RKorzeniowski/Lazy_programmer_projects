from util import getKaggleMNIST
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle

# siec 784 -> 350 -> 784


class AutoEncoder():
    def __init__(self, D, M):

        # input hidden
        Wi = tf.random_normal(shape=(D, M)) * 2 / np.sqrt(M)
        bi = np.zeros(M).astype(np.float32)

        # hidden output
        Wh = tf.random_normal(shape=(M, D)) * 2 / np.sqrt(D)
        bh = np.zeros(D).astype(np.float32)

        self.Wi = tf.Variable(Wi)
        self.Wh = tf.Variable(Wh)
        self.bi = tf.Variable(bi)
        self.bh = tf.Variable(bh)

        # placeholder for a batch of data
        self.X_in = tf.placeholder(tf.float32, shape=(None, D))
        self.X_hat = self.forward_output(self.X_in)

        logits = self.forward_logit(self.X_in)
        self.cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.X_in,
                logits=logits
            )
        )

        self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.cost)
        #self.train_op = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(self.cost)

    def fit(self, X, epochs=1, batch_sz=100, show_plot=False):
        N, D = X.shape

        n_batches = N // batch_sz

        costs = []
        print("trainnig autoencoder")
        for i in range(epochs):
            print("epoch: ", epochs)
            X = shuffle(X)
            for j in range(n_batches):
                batch = X[j * batch_sz:j * batch_sz + batch_sz]
                _, c = self.session.run((self.train_op, self.cost), feed_dict={self.X_in: batch})
                if j % 10:
                    print("j / n_batches:", j, "/", n_batches, "cost:", c)
                costs.append(c)
        if show_plot:
            plt.plot(costs)
            plt.show()

    def forward_hidden(self, X):
        # return tf.nn.sigmoid(tf.matmul(X, self.Wi) + self.bi)
        return tf.nn.relu(tf.matmul(X, self.Wi) + self.bi)
        # return tf.nn.tanh(tf.matmul(X, self.Wi) + self.bi)

    def forward_logit(self, X):  # just output without using sigmoid on it
        Z = self.forward_hidden(X)
        return tf.matmul(Z, self.Wh) + self.bh

    def forward_output(self, X):
        return tf.nn.sigmoid(self.forward_logit(X))

    def set_session(self, session):
        self.session = session

    def predict(self, X):
        return self.session.run(self.X_hat, feed_dict={self.X_in: X})


def single_AE():
    X, Y, Xtest, Ytest = getKaggleMNIST()

    _, D = X.shape

    autoencoder = AutoEncoder(D, 250)
    init_op = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init_op)
        autoencoder.set_session(session)
        autoencoder.fit(X)

        done = False
        while not done:
            i = np.random.choice(len(Xtest))
            x = Xtest[i]
            y = autoencoder.predict([x])

            plt.subplot(1, 2, 1)
            plt.imshow(x.reshape(28, 28), cmap='gray')
            plt.title('Original')

            plt.subplot(1, 2, 2)
            plt.imshow(y.reshape(28, 28), cmap='gray')
            plt.title('Reconstructed')

            plt.show()

            ans = input("Generate another?")
            if ans and ans[0] in ('n' or 'N'):
                done = True


if __name__ == '__main__':
    single_AE()
