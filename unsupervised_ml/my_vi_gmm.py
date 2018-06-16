import numpy as np
import matplotlib.pyplot as plt
from util import getKaggleMNIST
from sklearn.mixture import BayesianGaussianMixture
from scipy.stats import multivariate_normal as mvn


class BayesClassifier:
    def fit(self, X, Y):
        self.K = len(set(Y))

        # loop on all classes and all reprezentatives of this classes
        self.gaussians = []
        for k in range(self.K):
            Xk = X[Y == k]
            # max number of clusters
            gmm = BayesianGaussianMixture(10)
            gmm.fit(Xk)
            self.gaussians.append(gmm)

    # drawing sample for a given class y
    # finds a gaussian belonging to this class
    def sample_given_y(self, y):
        gmm = self.gaussians[y]
        sample = gmm.sample()
        # sample retrun 2 thing:
        # 1) sample
        # 2) cluster it came from
        mean = gmm.means_[sample[1]]
        return sample[0].reshape(28, 28), mean.reshape(28, 28)

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

        sample, mean = bay.sample_given_y(k)

        plt.subplot(1, 2, 1)
        plt.imshow(sample, cmap='gray')
        plt.title("sample")
        plt.subplot(1, 2, 2)
        plt.imshow(mean, cmap='gray')
        plt.title("mean")
        plt.show()

    # generate rdm sample
    sample, mean = bay.sample()
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


# def main():
#     X, Y, _, _ = getKaggleMNIST()

#     BGM = BayesianGaussianMixture()
#     k = 9
#     Xk = X[Y == k]
#     Yk = Y[Y == k]

#     BGM.fit(Xk, Yk)

#     sample_B = BGM.sample()
#     sample_B = sample_B[0]
#     print(sample_B[0].shape)
#     sample_B = sample_B[0].reshape((28, 28))
#     plt.imshow(sample_B, cmap='gray')
#     plt.title("rdm sample")
#     plt.show()


# if __name__ == '__main__':
#     main()
