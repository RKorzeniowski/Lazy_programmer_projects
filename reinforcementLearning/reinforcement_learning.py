import numpy as np
import matplotlib.pyplot as plt
# # we create a machine

# chyba cos zle jest w baysjanowym bo odbiega dosc mocno


class BayesianBandita:
  def __init__(self, true_mean):
    self.true_mean = true_mean
    # parameters for mu - prior is N(0,1)
    self.predicted_mean = 0
    self.lambda_ = 1
    self.sum_x = 0  # for convenience
    self.tau = 1

  def pull(self):
    return np.random.randn() + self.true_mean

  def sample(self):
    return np.random.randn() / np.sqrt(self.lambda_) + self.predicted_mean

  def update(self, x):
    self.lambda_ += self.tau
    self.sum_x += x
    self.predicted_mean = self.tau * self.sum_x / self.lambda_


class BayesianBandit:
  """
  m - true mean
  m0 - predicted mean
  parameters for mu - prior is N(0,1)
  """

  def __init__(self, m):
    self.m = m
    # parameters for mu - prior is N(0,1)
    self.m0 = 0
    self.lambda0 = 1
    self.sum_x = 0
    # mean of distribution (that we assume)
    self.tau = 1

  def pull(self):
    return np.random.randn() + self.m

  # here we sample a sample with gaussian mean m0 and precision lambda0
  def sample(self):
    return np.random.randn() / np.sqrt(self.lambda0) + self.m0

  def update(self, x):
    # assume tau = 1
    self.lambda0 += self.tau
    self.sum_x += x
    self.m0 = self.tau * self.sum_x / self.lambda0


class Bandit_UCB1(object):

  def __init__(self, m):
    self.mean = 10
    self.m = m
    self.N = 0  # + 10e-6
    # print(m)

  def pull(self):
    return np.random.randn() + self.m

  def update(self, x, allN):
    self.N += 1
    self.mean = (1 - 1.0 / self.N) * self.mean + 1.0 / self.N * x
    self.mean = self.mean + np.sqrt(2 * np.log(allN) / N)

    # calculated avg mean from old mean


class Bandit_optimistic(object):

  def __init__(self, m, upper_limit):
    self.mean = upper_limit
    self.m = m
    self.N = 1
    # print(m)

  def pull(self):
    return np.random.randn() + self.m

  def update(self, x):
    self.N += 1
    self.mean = (1 - 1.0 / self.N) * self.mean + 1.0 / self.N * x
    # calculated avg mean from old mean


class Bandit(object):

  def __init__(self, m):
    self.mean = 0
    self.m = m
    self.N = 0
    # print(m)

  def pull(self):
    return np.random.randn() + self.m

  def update(self, x):
    self.N += 1
    self.mean = (1 - 1.0 / self.N) * self.mean + 1.0 / self.N * x
    # calculated avg mean from old mean


def run_greedy_eps(m1, m2, m3, eps, N, decaying):
  # create list of machines
  bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]

  # bob = bandits[1]
  # joe = bandits[2]
  # print(bob.mean, joe.m)
  # print(np.argmax([bob.m, joe.m]))

  data = np.empty(N)

  for i in range(N):
    p = np.random.random()  # od 0 do 1 rdm number
    if decaying:
      eps = (1.0 / (i + 1))
    if p > eps:  # > in 90%
      j = np.argmax([b.mean for b in bandits])
    else:
      j = np.random.randint(3)
    x = bandits[j].pull()
    bandits[j].update(x)

    # create moving average
    data[i] = x

  # cumsum dodaje elementy do siebie 1: #1 -> 2: #1+#2 ...
  cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

  # plt.plot(cumulative_average)
  # plt.plot(np.ones(N) * m1)
  # plt.plot(np.ones(N) * m2)
  # plt.plot(np.ones(N) * m3)
  # plt.xscale('log')
  # plt.show()

  for b in bandits:
    print(b.mean)

  return cumulative_average


def run_optimistic(m1, m2, m3, eps, N, upper_limit):
  # create list of machines
  bandits = [Bandit_optimistic(m1, upper_limit), Bandit_optimistic(m2, upper_limit), Bandit_optimistic(m3, upper_limit)]

  # bob = bandits[1]
  # joe = bandits[2]
  # print(bob.mean, joe.m)
  # print(np.argmax([bob.m, joe.m]))

  data = np.empty(N)

  for i in range(N):
    # mean is too good so we check it to get the real mean (the first machine that stops falling is the best and we stop on it)
    j = np.argmax([b.mean for b in bandits])

    x = bandits[j].pull()
    bandits[j].update(x)

    # create moving average
    data[i] = x

  # cumsum dodaje elementy do siebie 1: #1 -> 2: #1+#2 ...
  cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

  # plt.plot(cumulative_average)
  # plt.plot(np.ones(N) * m1)
  # plt.plot(np.ones(N) * m2)
  # plt.plot(np.ones(N) * m3)
  # plt.xscale('log')
  # plt.show()

  for b in bandits:
    print(b.mean)

  return cumulative_average


def ucb(mean, n, nj):
  if nj == 0:
    return float('inf')
  return mean + np.sqrt(2 * np.log(n) / nj)


def run_UPB(m1, m2, m3, eps, N):
  # create list of machines
  bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]

  # bob = bandits[1]
  # joe = bandits[2]
  # print(bob.mean, joe.m)
  # print(np.argmax([bob.m, joe.m]))

  data = np.empty(N)

  for i in range(N):

    #N = N-1+10e-6

    #j = np.argmax([b.mean for b in bandits])

    # np.sqrt(2 * np.log(allN) / N)
    j = np.argmax([ucb(b.mean, i + 1, b.N) for b in bandits])

    x = bandits[j].pull()
    bandits[j].update(x)

    # create moving average
    data[i] = x

  # cumsum dodaje elementy do siebie 1: #1 -> 2: #1+#2 ...
  cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

  # plt.plot(cumulative_average)
  # plt.plot(np.ones(N) * m1)
  # plt.plot(np.ones(N) * m2)
  # plt.plot(np.ones(N) * m3)
  # plt.xscale('log')
  # plt.show()

  for b in bandits:
    print(b.mean)

  return cumulative_average


def run_baysian(m1, m2, m3, N):
  # create list of machines
  bandits = [BayesianBandita(m1), BayesianBandita(m2), BayesianBandita(m3)]

  # bob = bandits[1]
  # joe = bandits[2]
  # print(bob.mean, joe.m)
  # print(np.argmax([bob.m, joe.m]))

  data = np.empty(N)

  for i in range(N):
    # mean is too good so we check it to get the real mean (the first machine that stops falling is the best and we stop on it)
    j = np.argmax([b.sample() for b in bandits])

    x = bandits[j].pull()
    bandits[j].update(x)

    # create moving average
    data[i] = x

  # cumsum dodaje elementy do siebie 1: #1 -> 2: #1+#2 ...
  cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

  # plt.plot(cumulative_average)
  # plt.plot(np.ones(N) * m1)
  # plt.plot(np.ones(N) * m2)
  # plt.plot(np.ones(N) * m3)
  # plt.xscale('log')
  # plt.show()

  for b in bandits:
    print(b.predicted_mean)  # m0

  return cumulative_average


def run_experiment(m1, m2, m3, N):
  bandits = [BayesianBandit(m1), BayesianBandit(m2), BayesianBandit(m3)]

  data = np.empty(N)

  for i in range(N):
    # optimistic initial values
    j = np.argmax([b.sample() for b in bandits])
    x = bandits[j].pull()
    bandits[j].update(x)

    # for the plot
    data[i] = x
  cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

  # plot moving average ctr
  # plt.plot(cumulative_average)
  # plt.plot(np.ones(N)*m1)
  # plt.plot(np.ones(N)*m2)
  # plt.plot(np.ones(N)*m3)
  # plt.xscale('log')
  # plt.show()

  return cumulative_average


if __name__ == '__main__':
  N = 10000
  m1 = 2
  m2 = 3
  m3 = 5
  upper_limit = 10

  # opptimistic strat
  co_1 = run_optimistic(m1, m2, m3, 0.05, N, upper_limit)
  # c_01 = run_optimistic(m1, m2, m3, 0.01, N, upper_limit)
  # c_05 = run_optimistic(m1, m2, m3, 0.05, N, upper_limit)

  # eps greedy strat
  cp_1 = run_greedy_eps(m1, m2, m3, 0.05, N, True)  # decaying epsilon

  cU_1 = run_UPB(m1, m2, m3, 0.05, N)

  cB_1 = run_baysian(m1, m2, m3, N)
  cBL_1 = run_experiment(m1, m2, m3, N)
  run_experiment

  # # log scale plot
  # plt.plot(c_1, label='eps = 0.1')
  # plt.plot(c_05, label='eps = 0.05')
  # plt.plot(c_01, label='eps = 0.01')
  # plt.legend()
  # plt.xscale('log')
  # plt.show()

  # # linear plot
  # plt.plot(c_1, label='eps = 0.1')
  # plt.plot(c_05, label='eps = 0.05')
  # plt.plot(c_01, label='eps = 0.01')

  plt.plot(cp_1, label='eps = 0.1')
  plt.plot(co_1, label='opt = 0.1')
  plt.plot(cU_1, label='UCB = 0.1')
  plt.plot(cB_1, label='Bay = 0.1')
  plt.plot(cBL_1, label='BayL = 0.1')
  # plt.xscale('log')
  plt.legend()
  plt.show()
