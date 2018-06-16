import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn


def softplus(x):
    # log1p(x) = log(1+x). Use cuz its more numerically stable
    return np.log1p(np.exp(x))


# weights for neural network that is 4x3x2*2 decoder->codedstate->decoder

W1 = np.random.randn(4, 3)
W2 = np.random.randn(3, 2 * 2)

# why 2*2? We need 2 means and 2 std

# ignore bias for simplicity


def forward(X, W1, W2):
    hidden = np.tanh(X.dot(W1))
    output = hidden.dot(W2)
    mean = output[:2]
    stddev = softplus(output[2:])
    return mean, stddev


# make a random input
x = np.random.randn(4)

mean, stddev = forward(x, W1, W2)
print("mean: ", mean)
print("stddev: ", stddev)

# draw samples
samples = mvn.rvs(mean=mean, cov=stddev**2, size=10000)
# cov var along diaginal

# plot samples
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
plt.show()
