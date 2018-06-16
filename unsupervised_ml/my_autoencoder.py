import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import relu, error_rate, getKaggleMNIST, init_weights


def T_shared_zeros_like32(p):
    # p is a Theano shared itself
    return theano.shared(np.zeros_like(p.get_value(), dtype=np.float32))


def momentum_updates(cost, params, mu, learning_rate):
    # momentum changes
    dparams = [T_shared_zeros_like32(p) for p in params]

    updates = []
    grads = T.grad(cost, params)
    for p, dp, g in zip(params, dparams, grads):
        dp_update = mu * dp - learning_rate * g
        p_update = p + dp_update

        updates.append((dp, dp_update))
        updates.append((p, p_update))
    return updates


class AutoEncoder(object):

    def __init__(self, M, an_id):
        self.M = M
        self.id = an_id

    def fit(self, X, activation=relu, lr=0.5, epochs=1, mu=0.99, batch_sz=20, print_period=100, show_fig=False):
        # X = X.astype(np.float32)
        mu = np.float32(mu)
        lr = np.float32(lr)

        # init hidden layers
        N, D = X.shape
        n_batches = N // batch_sz

        # HiddenLayer could do this but i dont know whats up with the ids
        W0 = init_weights((D, self.M))
        self.W = theano.shared(W0, 'W_%s' % self.id)
        self.bh = theano.shared(np.zeros(self.M, dtype=np.float32), 'bh_%s' % self.id)
        self.bo = theano.shared(np.zeros(D, dtype=np.float32), 'bo_%s' % self.id)
        self.params = [self.W, self.bh, self.bo]
        self.forward_params = [self.W, self.bh]

        # shit for momentum
        # TODO: technically these should be reset before doing backprop
        self.dW = theano.shared(np.zeros(W0.shape), 'dW_%s' % self.id)
        self.dbh = theano.shared(np.zeros(self.M), 'dbh_%s' % self.id)
        self.dbo = theano.shared(np.zeros(D), 'dbo_%s' % self.id)
        self.dparams = [self.dW, self.dbh, self.dbo]
        self.forward_dparams = [self.dW, self.dbh]

        X_in = T.matrix('X_%s' % self.id)
        X_hat = self.forward_output(X_in)

        H = T.nnet.sigmoid(X_in.dot(self.W) + self.bh)
        self.hidden_op = theano.function(
            inputs=[X_in],
            outputs=H,
        )

        self.predict = theano.function(
            inputs=[X_in],
            outputs=X_hat,
        )

        # mse
        # cost = ((X_in - X_hat) * (X_in - X_hat)).sum() / N #mean or sum and mse as cost function

        # cross entropy
        cost = -(X_in * T.log(X_hat) + (1 - X_in) * T.log(1 - X_hat)).flatten().mean()
        cost_op = theano.function(
            inputs=[X_in],
            outputs=cost,
        )

        # grad descent + adding momentum changes
        updates = momentum_updates(cost, self.params, mu, lr)
        train_op = theano.function(
            inputs=[X_in],
            updates=updates,
        )

        costs = []
        print("training autoencoder: %s" % self.id)
        print("epochs to do:", epochs)
        for i in range(epochs):
            print("epoch:", i)
            X = shuffle(X)
            for j in range(n_batches):
                batch = X[j * batch_sz:(j * batch_sz + batch_sz)]
                train_op(batch)
                the_cost = cost_op(batch)  # technically we could also get the cost for Xtest here
                if j % 10 == 0:
                    print("j / n_batches:", j, "/", n_batches, "cost:", the_cost)
                costs.append(the_cost)
        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward_hidden(self, X):
        Z = T.nnet.sigmoid(X.dot(self.W) + self.bh)
        return Z

    def forward_output(self, X):
        Z = self.forward_hidden(X)
        Y = T.nnet.sigmoid(Z.dot(self.W.T) + self.bo)
        return Y


def main():
    pass


def test_single_autoencoder():
    Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()

    autoencoder = AutoEncoder(300, 0)
    autoencoder.fit(Xtrain, epochs=2, show_fig=True)

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
    test_single_autoencoder()
    # main()
