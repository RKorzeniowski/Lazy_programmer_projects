import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import init_weight, get_robert_frost, get_wikipedia_data


class SimpleRNN:
    def __init__(self, D, M, V):
        self.D = D
        self.M = M
        self.V = V

    def fit(self, X, learning_rate=10e-1, mu=0.99, reg=1.0, activation=T.tanh, epochs=500, show_fig=False):
        N = len(X)
        D = self.D
        M = self.M
        V = self.V

        # inital weights
        We = init_weight(V, D)
        Wx = init_weight(D, M)
        Wh = init_weight(M, M)
        bh = np.zeros(M)
        h0 = np.zeros(M)
        #z = np.ones(M)
        Wxz = init_weight(D, M)
        Whz = init_weight(M, M)
        bz = np.zeros(M)
        Wo = init_weight(M, V)
        bo = np.zeros(V)
        thX, thY, py_x, prediction = self.set(We, Wx, Wh, bh, h0, Wxz, Whz, bz, Wo, bo, activation)

        lr = T.scalar('lr')

        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value() * 0) for p in self.params]

        # updates = [
        #     (p, p + mu * dp - learning_rate * g) for p, dp, g in zip(self.params, dparams, grads)
        # ] + [
        #     (dp, mu * dp - learning_rate * g) for dp, g in zip(dparams, grads)
        # ]

        updates = []
        for p, dp, g in zip(self.params, dparams, grads):
            new_dp = mu * dp - lr * g
            updates.append((dp, new_dp))

            new_p = p + new_dp
            updates.append((p, new_p))

        self.predict_op = theano.function(inputs=[thX], outputs=prediction)
        self.train_op = theano.function(
            inputs=[thX, thY, lr],
            outputs=[cost, prediction],
            updates=updates,
        )

        costs = []
        n_total = sum((len(sentence) + 1) for sentence in X)
        for i in range(epochs):
            X = shuffle(X)
            n_correct = 0
            n_total = 0
            cost = 0
            for j in range(N):
                if np.random.random() < 0.1:
                    input_sequnece = [0] + X[j]
                    output_sequence = X[j] + [1]
                else:
                    input_sequnece = [0] + X[j][:-1]
                    output_sequence = X[j]
                n_total += len(output_sequence)

                c, p = self.train_op(input_sequnece, output_sequence, learning_rate)
                cost += c
                for pj, xj in zip(p, output_sequence):
                    if pj == xj:
                        n_correct += 1
            print("i:", i, "cost:", cost, "correct rate:", (float(n_correct) / n_total))
            if (i + 1) % 500 == 0:
                learning_rate /= 2
            costs.append(cost)

        if show_fig:
            plt.plot(costs)
            plt.show()

    def save(self, filename):
        np.savez(filename, *[p.get_value() for p in self.params])

    @staticmethod
    def load(filename, activation=T.nnet.relu):
        # saving activation is a good idea
        npz = np.load(filename)
        We = npz['arr_0']
        Wx = npz['arr_1']
        Wh = npz['arr_2']
        bh = npz['arr_3']
        h0 = npz['arr_4']
        Wxz = npz['arr_5']
        Whz = npz['arr_6']
        bz = npz['arr_7']
        Wo = npz['arr_8']
        bo = npz['arr_9']
        V, D = We.shape
        _, M = Wx.shape
        rnn = SimpleRNN(D, M, V)
        rnn.set(We, Wx, Wh, bh, h0, Wxz, Whz, bz, Wo, bo, activation)
        return rnn

    def set(self, We, Wx, Wh, bh, h0, Wxz, Whz, bz, Wo, bo, activation):
        self.f = activation

        # redundant - see how you can improve it
        self.We = theano.shared(We)
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wxz = theano.shared(Wxz)
        self.Whz = theano.shared(Whz)
        self.bz = theano.shared(bz)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)

        self.params = [self.We, self.Wx, self.Wh, self.h0, self.Wxz, self.Whz, self.bz, self.Wo, self.bo]

        thX = T.ivector('X')
        Ei = self.We[thX]  # T x D matrix
        thY = T.ivector('Y')

        def recurrence(x_t, h_t1):
            hhat_t = self.f(h_t1.dot(self.Wh) + self.bh + x_t.dot(self.Wx))  # chyba byl blad h_t = self.f(self.h_t1.dot(self.Wh) + self.bh + x_t.dot(self.Wx))
            z_t = T.nnet.sigmoid(x_t.dot(self.Wxz) + h_t1.dot(self.Whz) + self.bz)
            h_t = (1 - z_t) * h_t1 + z_t * hhat_t
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            return h_t, y_t

        [h, y], _ = theano.scan(
            fn=recurrence,
            outputs_info=[self.h0, None],  # initial stage for h but nothing for y
            sequences=Ei,
            n_steps=Ei.shape[0],
        )

        py_x = y[:, 0, :]
        prediction = T.argmax(py_x, axis=1)
        self.predict_op = theano.function(
            inputs=[thX],
            outputs=[py_x, prediction],
            allow_input_downcast=True,
        )
        return thX, thY, py_x, prediction

    def generate(self, word2idx):
        idx2word = {v: k for k, v in word2idx.items()}  # pewnie problems z iter
        V = len(word2idx)

        n_lines = 0

        X = [0]

        while n_lines < 4:
            # print "X:", X
            PY_X, _ = self.predict_op(X)[-1]
            PY_X, _ = PY_X[-1].flatten()
            P = [np.random.choice(V, p=PY_X)]
            X = np.concatenate([X, P])
            P = P[-1]
            if P > 1:
                # it's a real word, not start/end token
                word = idx2word[P]
                print(word, end=" ")
            elif P == 1:
                # end token
                n_lines += 1
                X = [0]
                print('')


def train_poetry():
    sentences, word2idx = get_robert_frost()
    rnn = SimpleRNN(50, 50, len(word2idx))
    rnn.fit(sentences, learning_rate=10e-5, show_fig=True, activation=T.nnet.relu, epochs=2000)
    rnn.save('RRNN_D50_M50_epochs2000_relu_my.npz')


def generate_potery():
    sentences, word2idx = get_robert_frost()
    rnn = SimpleRNN.load('RRNN_D50_M50_epochs2000_relu_my.npz')  # sprawdzic czy dziala tez bez
    #rnn = SimpleRNN.load('RNN_D30_M30_epochs2000_relu_my', T.nnet.relu)
    rnn.generate(word2idx)


if __name__ == '__main__':
    train_poetry()
    generate_potery()
