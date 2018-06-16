import numpy as np
import theano
import theano.tensor as T

from util import init_weight


class LSTM:
    """docstring for LSTM"""

    def __init__(self, Mi, Mo, activation):
        self.Mi = Mi
        self.Mo = Mo
        self.f = activation

        # input gate
        self.Wxi = init_weight(Mi, Mo)
        self.Whi = init_weight(Mo, Mo)
        self.Wci = init_weight(Mo, Mo)
        self.bi = np.zeros(Mi)

        # forget gate
        self.Wxf = init_weight(Mi, Mo)
        self.Whf = init_weight(Mo, Mo)
        self.Wcf = init_weight(Mo, Mo)
        self.bf = np.zeros(Mo)

        # candidate cell
        self.Wxc = init_weight(Mi, Mo)
        self.Whc = init_weight(Mo, Mo)
        self.bc = np.zeros(Mo)

        # output gate
        self.Wxo = init_weight(Mi, Mo)
        self.Who = init_weight(Mo, Mo)
        self.Wco = init_weight(Mo, Mo)
        self.bo = np.zeros(Mo)

        # initial state of h and c
        self.h0 = np.zeros(Mo)
        self.c0 = np.zeros(Mo)  # czy to dobry rozmiar

        # initialize in theano
        # input gate
        self.Wxi = theano.shared(Wxi)
        self.Whi = theano.shared(Whi)
        self.Wci = theano.shared(Wci)
        self.bi = theano.shared(bi)

        # forget gate
        self.Wxf = theano.shared(Wxf)
        self.Whf = theano.shared(Whf)
        self.Wcf = theano.shared(Wcf)
        self.bf = theano.shared(bf)

        # candidate gate
        self.Wxc = theano.shared(Wxc)
        self.Whc = theano.shared(Whc)
        self.bc = theano.shared(bc)

        # output gate
        self.Wxo = theano.shared(Wxo)
        self.Who = theano.shared(Who)
        self.Wco = theano.shared(Wco)
        self.bo = theano.shared(bo)

        # initial states
        self.h0 = theano.shared(h0)
        self.c0 = theano.shared(c0)

        # list for grad update
        params = [self.Wxi, self.Whi, self.Wci, self.bi, self.Wxf, self.Whf, self.Wcf, self.bf, self.Wxc, self.Whc, self.bc, self.Wxo, self.Who, self.Wco, self.bo, self.h0, self.c0]

    def recurance(self, x_t, h_t1, c_t1):
        # input gate
        i_t = T.nnet.sigmoid(x_t.dot(self.Wxi) + h_t1.dot(self.Whi) + c_t1.dot(self.Wci) + self.bi)
        # forget gate
        f_t = T.nnet.sigmoid(x_t.dot(self.Wxf) + h_t1.dot(self.Whf) + c_t1.dot(self.Wcf) + self.bf)
        # candidate gate
        c_t = f_t * c_t1 + i_t * T.tanh(x_t.dot(self.Wxc) + h_t1.dot(self.Whc) + self.bc)
        # output gate
        o_t = T.nnet.sigmoid(x_t.dot(self.Wxo) + h_t1.dot(self.Who) + c_t.dot(self.Wco) + self.bo)
        # state at step t
        h_t = o_t * T.tanh(c_t)
        return h_t, c_t

    def output(self, x):

        [h, c], _ = theano.scan(
            fn=self.recurance,
            output_info=[self.h0, self.c0],
            sequence=x,
            n_steps=x.shape[0],
        )
        return h
