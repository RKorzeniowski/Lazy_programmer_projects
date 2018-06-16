import numpy as np
import theano
import theano.tensor as T

from sklearn.utils import shuffle
from util import init_weight


class GRU():
    """docstring for GRU
    Mi -> inputs size
    Mo -> outputs size
    """

    def __init__(self, Mi, Mo, actication):
        self.Mi = Mi
        self.Mo = Mo
        self.f = actication

        Wxr = init_weight(Mi, Mo)  # input to the reset gate
        Whr = init_weight(Mi, Mo)  # hidden to the reset gate
        br = np.zeros(Mi)  # bias to the reset gate
        Wxz = init_weight(Mi, Mo)  # input to the update gate
        Whz = init_weight(Mo, Mo)  # hidden to update gate
        bz = np.zeros(Mo)  # bias for the update gate
        Wxh = init_weight(Mi, Mo)  # input to hidden
        Whh = init_weight(Mo, Mo)  # hidden to hidden
        bh = np.zeros(Mo)  # bias to hidden
        h0 = np.zeros(Mo)  # initial hidden state

        # create theano variables
        self.Wxr = theano.shared(Wxr)
        self.Whr = theano.shared(Whr)
        self.br = theano.shared(br)
        self.Wxz = theano.shared(Wxz)
        self.Whz = theano.shared(Whz)
        self.bz = theano.shared(bz)
        self.Wxh = theano.shared(Wxh)
        self.Whh = theano.shared(Whh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.params = [self.Wxr, self.Whr, self.br, self.Wxz, self.Whz, self.bz, self.Wxh, self.Whh, self.bh]

    def recurrance(self, x_t, h_t1):
        # reset gate
        r = T.nnet.sigmoid(x_t.dot(self.Wxr) + h_t1.dot(self.Whr) + self.br)
        # update gate
        z = T.nnet.sigmoid(x_t.dot(self.Wxz) + h_t1.dot(self.Whz) + self.bz)
        # * is element by element multipication
        hhat_t = self.f(x_t.dot(self.Wxh) + (r * h_t1).dot(self.Whh) + self.bh)
        h = (1 - z) * h_t1 + z * hhat_t
        return h

    def output(self, x):
        h, _ = theano.scan(
            fn=self.recurrence,
            output_info=[self.h0],
            seqences=x,
            n_setps=x.shape[0],
        )
        return h
