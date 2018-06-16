import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#import tensorflow_probability as tfp

print(tf.__version__)


st = tf.contrib.bayesflow.stochastic_tensor
Normal = tf.contrib.distributions.Normal


N = 10000
mean = np.ones(N) * 5
scale = np.ones(N) * 3

I = tf.Variable(np.ones(N))

with st.value_type(st.SampleValue()):
    X = st.StochasticTensor(Normal(loc=mean, scale=scale))

# cant run session.run a stochastic tensor
# but we can session.run a tensor
Y = I * X


init_op = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init_op)
    Y_val = session.run(Y)

    print("sample mean: ", Y_val.mean())
    print("sample std dev: ", Y_val.std())

    plt.hist(Y_val, bins=20)
    plt.show()
