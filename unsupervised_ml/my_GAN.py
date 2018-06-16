import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from util import getKaggleMNIST

# based on what generator generates? he starts random and then based on cost of distriminator gets better? no cuz it needs some paramters to generate different pictures per sample but we only go for is it fake or not sooo....

# cost function how it was defined? ones again


# 1st network
class Generator(object):
    def __init__():
        #   poj&reshape deconv5x5  deconv5x5  deconv5x5   deconv5x5
        # 100 -> 4x4x1024 -> 8x8x512 -> 16x16x256 ->32x32x128 -> 64x64x3

        pass

    def forward_train(self, X):  # use relu # batch norm
        # keep track of:
        #   X_b = next batch of data
        #   mu_b = mean(X_b)
        #   std_b = std(X_b)
        #   Y_b = (X_b - mu_b) / std_b
        #
        # we need to store current mu of data
        #
        # normalization before passing to activation function

        Z = tf.matmul(X, self.W) + b

        return self.f()

    def sample():
        pass

# 2nd network


class Discriminator(object):
    def __init__():
        #      conv5x5      conv5x5     conv5x5    conv5x5      full
        # 64x64x3 -> 32x32x64 -> 16x16x128 -> 8x8x256 -> 4x4x512 -> 1

        pass

    def forward_train():  # use leaky relu # batch norm
        # 1)through the layer
        pass
        # 2)normalize
        # 3)multiply

# where do i define 2 optimizers and costs? In classes foward functions(i dont think so) or in common loop in main ?


# flow in the GAN
# 1) z is generated form a uniform distrubiution
# 2) generator creates sth that is the same shape as training data = X_hat
# 3) some data from actual traning data = X
# 4) discriminator takes 2) & 3) dedides which output a decision (i guess access 0 or 1 )
# 5) binary cross entropy function and generator cost function based on 4)

def main():
    # load data
    Xtrain, _, Xtest, _ = getKaggleMNIST()

    # initialize generator network
    G = Generator()
    # initialize discriminator network
    D = Discriminator()

    # J = - [tlogy + (1-t)log(1-y)]
    #
    # J_D = - [ log(D(x)) + log(1-D(G(z))) ]
    #
    # for a batch J_D = - [ sum_x{log(D(x))} + sum_z{log(1-D(G(z)))} ]
    #
    # D(x) <- y = p(image is real | image ) = D(x; Theta_D), where p(image is real | image ) takes values between 0 and 1
    #   x = real images
    #   x_hat = fake images
    #
    # G(x) <- output from a generator
    #   we can get sample by following this 2 steps: (z is a latent pior)
    #   1) z ~ p(z) (same as with VAE... so check out how it was implemented there)
    #   2) x_hat = G(z)   (parameters of G are going to be reprezented as Theta_G; G(z;Theta_G))

    # (second term of the J_D equation D(G(z)) is very close to 0 and 1st term of a gradient is 0 wrt Theta_G so we get almost no update to parameters)
    # we flip the target the solve the problem  fake is now = 1 and true = 0
    # J_G = -E{logD(G(z))}        then generator cost and discriminator cost no longer sum to 0
    #

    cost_G = ...  # ??? 2 functions dJ/dD and dJ/dG or wat. no just 1 cost. no 2 costs
    cost_D =  # 1 how to get expected value? 2 how to get G(z) |i guess its just a sample from generator| 3how to get D() of sth

    learning_rate_G = 1e-3
    learning_rate_D = 1e-3

    self.train_op_G = tf.train.AdamOptimizer(learning_rate_G).minimize(self.cost_G)
    self.train_op_D = tf.train.AdamOptimizer(learning_rate_D).minimize(self.cost_D)

    self.init_op = tf.global_variables_initializer()  # this is wrong cuz we need to initialize Disc 1 time and run it on 2 datasets
    self.sess = tf.InteractiveSession()
    self.sess.run(self.init_op)

    n_batch = len(Xtrain) // batch_sz

    costs = []
    # create training loop
    for i in range(epochs):
        np.random.shuffle(Xtrain)
        print("epoch: ", i)
        for j in range(n_batch):
            # create batch of real images
            X = Xtrain[j * batch_sz:(j + 1) * batch_sz]
            # create test batch (well im not sure about that) but how elese it would be able to predicted images in a sample
            # generate batch of fake images (needs arguments like batch size, ...)
            X_hat = G.sample(batch_sz)  # i would give Y so the network knows what to predict
            # update parameters of a discriminator
            # definte it earlierTheta_D -= learning_rate_D * dJ / dTheta_D  # | has to be done by tf function
            # update parameters of a generator
            # definte it earlier Theta_G -= learning_rate_G * dJ / dTheta_G  # | has to be done by tf function
            # does the run give output
            _, c_D = self.sess.run((self.train_op_D, self.cost_D), feed_dict={self.X: X, self.X_hat: X_hat})  # i guess cost of distiminator is negative cost of generator
            _, c_G = self.sess.run((self.train_op_G, self.cost_G), feed_dict={})
            print("batch: ", j, " out of ", n_batch, ", cost ", c)
            costs.append(c)
    if plot:
        plt.plot(costs)
        plt.show()


if __name__ == '__main__':
    main()
