import numpy as np
import matplotlib.pyplot as plt

# distance function


def get_simple_data():
    # assume 3 means
    D = 2  # so we can visualize it more easily
    s = 4  # separation so we can control how far apart the means are
    mu1 = np.array([0, 0])
    mu2 = np.array([s, s])
    mu3 = np.array([0, s])

    N = 900  # number of samples
    X = np.zeros((N, D))
    X[:300, :] = np.random.randn(300, D) + mu1
    X[300:600, :] = np.random.randn(300, D) + mu2
    X[600:, :] = np.random.randn(300, D) + mu3

    return X


def d(u, v):
    diff = u - v
    return diff.dot(diff)  # square distance

# cost function


def cost(X, R, M):
    cost = 0
    for k in range(len(M)):
        for n in range(len(X)):
            cost += R[n, k] * d(M[k], X[n])
        return cost


def plot_k_means(X, K, max_iter=20, beta=1.0, show_plots=True):
    N, D = X.shape
    M = np.zeros((K, D))
    # R = np.zeros((N, K))
    exponents = np.empty((N, K))

    # initialize M to random
    for k in range(K):
        M[k] = X[np.random.choice(N)]

    costs = np.zeros(max_iter)
    for i in range(max_iter):
        # step 1: determine assignments / resposibilities
        # is this inefficient?
        for k in range(K):
            for n in range(N):
                # R[n,k] = np.exp(-beta*d(M[k], X[n])) / np.sum( np.exp(-beta*d(M[j], X[n])) for j in range(K) )
                exponents[n, k] = np.exp(-beta * d(M[k], X[n]))

        R = exponents / exponents.sum(axis=1, keepdims=True)
        # assert(np.abs(R - R2).sum() < 1e-10)

        # step 2: recalculate means
        for k in range(K):
            M[k] = R[:, k].dot(X) / R[:, k].sum()

        costs[i] = cost(X, R, M)
        if i > 0:
            if np.abs(costs[i] - costs[i - 1]) < 1e-5:
                break

    if show_plots:
        plt.plot(costs)
        plt.title("Costs")
        plt.show()

        random_colors = np.random.random((K, 3))
        colors = R.dot(random_colors)
        plt.scatter(X[:, 0], X[:, 1], c=colors)
        plt.show()

    return M, R

# def plot_k_means(X, K, max_iter=20, beta=1.0):
#     N, D = X.shape
#     M = np.zeros((K, D))  # means initialiazied as 0
#     R = np.zeros((N, K))  # resposibility matrix

#     # initialized means to some random points x_n
#     for k in range(K):
#         M[k] = X[np.random.choice(N)]

#     grid_w = 5
#     grid_h = max_iter / grid_w
#     random_colors = np.random.random((K, 3))
#     plt.figure()

#     costs = np.zeros(max_iter)
#     for i in range(max_iter):

#         colors = R.dot(random_colors)
#         plt.subplot(grid_w, grid_h, i + 1)
#         plt.scatter(X[:, 0], X[:, 1], c=colors)

#         # step one. Set R (R is a factor of how important in calculating mean has this point. Its a measure of how sure are we that this point belongs to this distribution. Its bettwend 0 and 1)
#         for k in range(K):
#             for n in range(N):
#                 # d(M[k], X[n]) is a distance from mean_k to point x_n
#                 R[n, k] = np.exp(-beta * d(M[k], X[n])) / np.sum(np.exp(-beta * d(M[j], X[n])) for j in range(K))

#         # step two. Recalculate the means
#         for k in range(K):
#             M[k] = R[:, k].dot(X) / R[:, k].sum()

#         # calculate the cost
#         costs[i] = cost(X, R, M)
#         if i > 0:
#             if np.abs(costs[i] - costs[i - 1]) < 0.1:
#                 break

# plot cost
# plt.plot(costs)
# plt.title("costs")
# plt.show()

# plot clusters
#random_colors = np.random.random((K, 3))
#colors = R.dot(random_colors)
#plt.scatter(X[:, 0], X[:, 1], c=colors)
#    plt.show()


def main():
    D = 2
    s = 4
    mu1 = np.array([0, 0])
    mu2 = np.array([s, s])
    mu3 = np.array([0, s])

    N = 900
    X = np.zeros((N, D))
    X[:300, :] = np.random.randn(300, D) + mu1
    X[300:600, :] = np.random.randn(300, D) + mu2
    X[600:, :] = np.random.randn(300, D) + mu3

    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    K = 3
    plot_k_means(X, K)

    K = 5
    plot_k_means(X, K, max_iter=30)

    K = 5
    plot_k_means(X, K, max_iter=30, beta=0.3)


if __name__ == '__main__':
    main()
