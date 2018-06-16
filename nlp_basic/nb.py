from __future__ import print_function, division
from builtins import range

from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

data = pd.read_csv('spambase/spambase.data').as_matrix() # use pandas for convenience
np.random.shuffle(data) # shuffle each row in-place, but preserve the row

X = data[:, :48]
Y = data[:, -1]

Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[:-100,]
Ytest = Y[:-100,]

model = MultinomialNB()
model.fit(Xtrain, Ytrain)

print("Classification rate for NB:", model.score(Xtest, Ytest))

from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier()
model.fit(Xtrain, Ytrain)
print("Classification rate for AdaBoost:", model.score(Xtest, Ytest))