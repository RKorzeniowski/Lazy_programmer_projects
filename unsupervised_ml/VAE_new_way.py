import tensorflow as tf
import tensorflow_probability as tfp

# Assumes user supplies `likelihood`, `prior`, `surrogate_posterior`
# functions and that each returns a
# tf.distribution.Distribution-like object.
elbo_loss = tfp.vi.monte_carlo_csiszar_f_divergence(
    f=tfp.vi.kl_reverse,  # Equivalent to "Evidence Lower BOund"
    p_log_prob=lambda z: likelihood(z).log_prob(x) + prior().log_prob(z),
    q=surrogate_posterior(x),
    num_draws=1)

train = tf.train.AdamOptimizer(
    learning_rate=0.01).minimize(elbo_loss)
