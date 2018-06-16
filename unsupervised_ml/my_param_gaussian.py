import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn


def softplus(x):
    return np.log1p(np.exp(x))

# we're going to make a neural network
# with the layer sizes (4, 3, 2)
# like a toy version of a decoder


W1 = np.random.randn(4, 3)
W2 = np.random.randn(3, 2 * 2)

# why 2 * 2?
# we need 2 components for the mean,
# and 2 components for the standard deviation!
# ignore bias terms for simplicity.


def forward(x, W1, W2):
    hidden = np.tanh(x.dot(W1))
    output = hidden.dot(W2)  # no activation
    mean = output[:2]
    stddev = softplus(output[2:])
    return mean, stddev


x = np.random.randn(4)

mean, stddev = forward(x, W1, W2)
print("mean:", mean)
print("stddev:", stddev)

samples = mvn.rvs(mean=mean, cov=stddev**2, size=10000)

plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
plt.show()
