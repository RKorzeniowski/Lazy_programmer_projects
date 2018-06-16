import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

Nclass = 500
D = 2  # dimentinality of input
M = 3  # hidden layer size
K = 3  # number of classes

# create classes , the data
X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
X = np.vstack([X1, X2, X3])

# create lables
Y = np.array([0] * Nclass + [1] * Nclass + [2] * Nclass)

N = len(Y)
T = np.zeros((N, K))
for i in range(N):
    T[i, Y[i]] = 1

# This vars will be inittialized by sess = tf.Session()


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def forward(X, W1, b1, W2, b2):
    Z = tf.sigmoid(tf.matmul(X, W1) + b1)
    return tf.matmul(Z, W2) + b2  # return activation not the output of the softmax


# its creates a grapth so it know how to calculate everything but it does not have value its just a placeholder
tfX = tf.placeholder(tf.float32, [None, D])
tfY = tf.placeholder(tf.float32, [None, K])

W1 = init_weights([D, M])
b1 = init_weights([M])
W2 = init_weights([M, K])
b2 = init_weights([K])

py_x = forward(tfX, W1, b1, W2, b2)
#logits = forward(tfX, W1, b1, W2, b2)


# Nx1 matrix of targets tfY
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=tfY))
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tfY, logits=logits))
# calculates the gradient automaticly

train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# initialiazie session and all the varaible that were defined earlier
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

# feed_dict= dictionary with tfplaceholders and the values of the acculat values you want to pass to this placeholders
for i in range(1000):
    sess.run(train_op, feed_dict={tfX: X, tfY: T})
    pred = sess.run(predict_op, feed_dict={tfX: X, tfY: T})
    if i % 10 == 0:
        print(np.mean(Y == pred))
