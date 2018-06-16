import numpy as np
import matplotlib.pyplot as plt
from my_MC_control import max_dict
from my_td0_pred import random_action
from iterative_policy import print_values, print_policy
from my_grid_word import standard_grid, negative_grid
from my_td0_pred import play_game, SMALL_ENOUGH, GAMMA, ALPHA, ALL_POSSIBLE_ACTIONS


#cos jest nie tak

class Model:
    def __init__(self):
        self.theta = np.random.randn(4) / 2

    def s2x(self, s):
        return np.array([s[0] - 1, s[1] - .5, s[0] * s[1] - 3, 1])

    def predict(self, s):
        x = self.s2x(s)
        return self.theta.dot(x)

    def grad(self, s):
        return self.s2x(s)


if __name__ == '__main__':

    grid = standard_grid()

    # print rewards
    print("rewards:")
    print_values(grid.rewards, grid)

    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'U',
        (2, 1): 'L',
        (2, 2): 'U',
        (2, 3): 'L',
    }

    model = Model()
    deltas = []

    N = 20000
    k = 1.0
    for it in range(N):
        if it % 10 == 0:
            k += 0.01
        alpha = ALPHA / k
        biggest_change = 0

    states_and_rewards = play_game(grid, policy)

    for t in range(len(states_and_rewards) - 1):
        s, _ = states_and_rewards[t]
        s2, r = states_and_rewards[t + 1]
        old_theta = model.theta.copy()
        if grid.is_terminal(s2):
            target = r
        else:
            target = r + GAMMA * model.predict(s2)
        model.theta += alpha * (target - model.predict(s)) * model.grad(s)
        biggest_change = max(biggest_change, np.abs(old_theta - model.theta).sum())
    deltas.append(biggest_change)

    plt.plot(deltas)
    plt.show()

    V = {}
    states = grid.all_states()

    for s in states:
        if s in grid.actions:
            V[s] = model.predict(s)
        else:
            V[s] = 0

    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)
