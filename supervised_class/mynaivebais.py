import numpy as np
from util import get_data
from datetime import datetime
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

# we asssume that cov = 0 and var = sd**2 = 1


class NaiveBayes(object):
    def fit(self, X, Y, smoothing=1e-2):
        self.gaussians = dict()  # we use gaussian to code pixel light intensity distribution (in data from 0 to 255)
        self.priors = dict()
        labels = set(Y)  # each type of char 'val' in column Y
        print(labels)
        for c in labels:
            current_x = X[Y == c]  # take only indexes of class that c is equal to now
            # print(current_x)
            self.gaussians[c] = {
                'mean': current_x.mean(axis=0),  # now that we have only one class in matrix we just take the mean
                'var': current_x.var(axis=0) + smoothing,  # same with variance
            }
            self.priors[c] = float(len(Y[Y == c])) / len(Y)  # number of occurances of the class now in Y / len(Y) is all classes

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

    # what we are doing is p(X|C)=p(pixel1|C)*p(pixel2|C)...p(pixel784|C)=N(x1;miu1,var1)*N(x2;miu2,var2)...N(x784;miu784,var784)
    def predict(self, X):
        N, D = X.shape  # cov offdiaiagonals are 0 so we need only array of D.
        K = len(self.gaussians)  # number of categories
        # print(K)
        P = np.zeros((N, K))
        for c, g in dict.items(self.gaussians):
            mean, var = g['mean'], g['var']  # does this gaussian has D dementions? i thnik yes
            P[:, c] = mvn.logpdf(X, mean=mean, cov=var) + np.log(self.priors[c])  # based on var and mean of some distribution i give probability that this data belongs to this dristribution (here x and miu are vectors so we use covariance matrix) |||log probability argmax{P(X|C)P(C)}=argmax{log P(X|C) + log P(C)}
            print(P)
        return np.argmax(P, axis=1)  # which of the classes has the highest probability


if __name__ == '__main__':
    X, Y = get_data(10000)
    Ntrain = len(Y) // 2
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    model = NaiveBayes()
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print("traning time:", (datetime.now() - t0))

    t0 = datetime.now()
    print("trainig acc", model.score(Xtrain, Ytrain))
    print("time to score", (datetime.now() - t0), "train lenght", len(Ytrain))

    t0 = datetime.now()
    print("test acc", model.score(Xtrain, Ytrain))
    print("time to score", (datetime.now() - t0), "test lenght", len(Ytrain))
