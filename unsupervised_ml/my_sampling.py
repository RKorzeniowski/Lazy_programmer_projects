from util import getKaggleMNIST
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import multivariate_normal as mvn


class BayesClassifier:
    def fit(self, X, Y):
        self.K = len(set(Y))

        # loop on all classes and all reprezentatives of this classes
        self.gaussians = []
        for k in range(self.K):
            Xk = X[Y == k]
            # Xk[x] is a image of number, Xk  is a list of all this images

            # axis 0 is summing down, so it sums average of a pixel brightness for all reprezentatives of this class
            mean = Xk.mean(axis=0)
            cov = np.cov(Xk.T)

            # dict that contains mean and cov of a sample
            #plt.imshow(mean.reshape((28, 28)), cmap="gray")
            # plt.show()
            #img = mvn.rvs(mean, cov)
            # plt.plot(img)
            # plt.show()
            g = {"m": mean, "c": cov}
            self.gaussians.append(g)

    # drawing sample for a given class y
    # finds a gaussian belonging to this class
    def sample_given_y(self, y):
        g = self.gaussians[y]
        return mvn.rvs(mean=g['m'], cov=g['c'])

    # draw sample from any class
    # it assumes that they are uniformly distrubuted (randint)
    def sample(self):
        y = np.random.randint(self.K)
        return self.sample_given_y(y)


def main():
    Xtrain, Ytrain, _, _ = getKaggleMNIST()

    bay = BayesClassifier()
    bay.fit(Xtrain, Ytrain)
    # = bay.sample()

    for k in range(bay.K):

        sample = bay.sample_given_y(k).reshape(28, 28)
        mean = bay.gaussians[k]['m'].reshape(28, 28)

        plt.subplot(1, 2, 1)
        plt.imshow(sample, cmap='gray')
        plt.title("sample")
        plt.subplot(1, 2, 2)
        plt.imshow(mean, cmap='gray')
        plt.title("mean")
        plt.show()

    # generate rdm sample
    sample = bay.sample().reshape(28, 28)
    plt.imshow(sample, cmap='gray')
    plt.title("rdm sample")
    plt.show()

    # train = pd.read_csv(path).as_matrix().astype(np.float32)

    # for k=1 which is Ytrain=1 we get number of rows which we use to pick rows from Xtrain

    # plot class1
    # indexes of all class 1 mudafuckers
    # num_class = 1
    # idx = [x for x in range(len(Ytrain)) if Ytrain[x] == num_class]

    # print(Xtrain[8].mean())

    # for x in Ytrain:
    #     print(x)
    # print(class1)
    # print(class1.mean())
    # print(np.cov(class1))
    # gauss = multivariate_normal.rvs(class1.mean(), np.cov(class1))
    # plt.plot(gauss)
    # plt.show()

    # plot distr


if __name__ == '__main__':
    main()
