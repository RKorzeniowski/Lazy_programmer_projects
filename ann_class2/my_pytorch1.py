import numpy as np
import matplotlib.pyplot as plt
from util import get_normalized_data

import torch
from torch.autograd import Variable
from torch import optim


X, Y = get_normalized_data()

_, D = X.shape
K = len(set(Y))

Xtrain = X[:-1000, ]
Ytrain = Y[:-1000]
Xtest = X[-1000:, ]
Ytest = Y[-1000:]

Ytrain2 = np.copy(Ytrain)

model = torch.nn.Sequential()

# ANN with layers [784] -> [500] -> [300] -> [10]
model.add_module("dense1", torch.nn.Linear(D, 500))
model.add_module("relu1", torch.nn.ReLU())
model.add_module("dense2", torch.nn.Linear(500, 300))
model.add_module("relu2", torch.nn.ReLU())
model.add_module("dense3", torch.nn.Linear(300, K))
# Note: no final softmax!
# just like Tensorflow, it's included in cross-entropy function

# http://pytorch.org/docs/master/nn.html#loss-functions
loss = torch.nn.CrossEntropyLoss(size_average=True)

# http://pytorch.org/docs/master/optim.html
optimizer = optim.Adam(model.parameters())


def train(model, loss, optimizer, inputs, labels):
    inputs = Variable(inputs, requires_grad=False)
    labels = Variable(labels, requires_grad=False)

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    logits = model.forward(inputs)
    output = loss.forward(logits, labels)

    # Backward
    output.backward()

    # Update paramteters
    optimizer.step()
    return output.data[0]

# define the prediction procedure
# also encapsulate these steps
# Note: inputs is a torch tensor


def score(model, inputs, labels):
    prediction = predict(model, inputs)
    return np.mean(labels.numpy() == prediction)


def get_cost(model, loss, inputs, labels):
    inputs = Variable(inputs, requires_grad=False)
    labels = Variable(labels, requires_grad=False)

    logits = model.forward(inputs)
    output = loss.forward(logits, labels)

    return output.data[0]


def predict(model, inputs):
    inputs = Variable(inputs, requires_grad=False)
    logits = model.forward(inputs)
    return logits.data.numpy().argmax(axis=1)


# convert the data arrays into torch tensors
Xtrain = torch.from_numpy(Xtrain).float()
Ytrain = torch.from_numpy(Ytrain).long()
Xtest = torch.from_numpy(Xtest).float()
Ytest = torch.from_numpy(Ytest).long()
#Ytest_ass = torch.from_numpy(Ytest).long()

epochs = 15
batch_size = 32
n_batches = Xtrain.size()[0] // batch_size

costs = []
test_costs = []
traning_accuracies = []
test_accuracies = []
for i in range(epochs):
    cost = 0
    cost_test = 0
    for j in range(n_batches):
        Xbatch = Xtrain[j * batch_size:(j + 1) * batch_size]
        Ybatch = Ytrain[j * batch_size:(j + 1) * batch_size]
        cost += train(model, loss, optimizer, Xbatch, Ybatch)

    #cost_test = get_cost(model, loss, Xtest, Ytest_ass)
    cost_test = get_cost(model, loss, Xtest, Ytest)

    Ypred = predict(model, Xtest)
    Ypred_train = predict(model, Xtrain)

    acc = score(model, Xtrain, Ytrain)
    acc_train = score(model, Xtest, Ytest)

    #acc = np.mean(Ytest == Ypred)
    #acc_train = np.mean(Ytrain2 == Ypred_train)
    print("Epoch: %d, cost: %f, acc: %.2f" % (i, cost / n_batches, acc))

    # plotting stuff
    costs.append(cost / n_batches)
    traning_accuracies.append(acc_train)
    test_accuracies.append(acc)
    test_costs.append(cost_test)


# EXERCISE: plot test cost + training accuracy too


# plot the results
plt.plot(costs)
plt.title('Training cost')
plt.show()

plt.plot(test_accuracies)
plt.title('Test accuracies')
plt.show()

plt.plot(test_costs)
plt.title('Test cost')
plt.show()

plt.plot(traning_accuracies)
plt.title('Traning accuracies')
plt.show()
