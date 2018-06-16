import numpy as np
from sortedcontainers import SortedList

from util import get_data
from datetime import datetime


class KNN(object):
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def preditc(self, X):
        y = np.zeros(len(X))
        for i, x in enumerate(X):
            sl = SortedList()
            for j, xt in enumerate(self.X):
                diff = x - xt
                d = diff.dot(diff)  # squared distance
                if len(sl) < self.k:
                    sl.add((d, self.y[j]))
                else:
                    if d < sl[-1][0]:
                        del sl[-1]
                        sl.add((d, self.y[j]))

            votes = {}  # dicts for votes to which class the point belongs
            for _, v in sl:
                votes[v] = votes.get(v, 0) + 1  # we flip the votes
            max_votes = 0
            max_votes_class = -1
            for v, count in dict.items(votes):
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v

            y[i] = max_votes_class
        return y

    def score(self, X, Y):
        P = self.preditc(X)
        return np.mean(P == Y)


if __name__ == '__main__':
    X, Y = get_data(2000)
    Ntrain = 1000
    Xtrain, Ytrain = X[: Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[:Ntrain:]
    for k in (1, 2, 3, 4, 5):
        knn = KNN(k)
        t0 = datetime.now()
        knn.fit(Xtrain, Ytrain)
        print("traning time:", (datetime.now() - t0))

        t0 = datetime.now()
        print("train accuracy:", knn.score(Xtrain, Ytrain))
        print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Ytrain))

        print("test accuracy:", knn.score(Xtrain, Ytrain))
        print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Ytest))
