import numpy as np
import matplotlib.pyplot as plt
from my_MC_control import max_dict
from my_td0_pred import random_action
from iterative_policy import print_values, print_policy
from my_grid_word import standard_grid, negative_grid
from my_MC_rnd import random_action, play_game, SMALL_ENOUGH, GAMMA

LEARNING_RATE = 0.001

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

    theta = np.random.randn(4) / 2

    def s2x(s):
        return np.array([s[0] - 1, s[1] - .5, s[0] * s[1] - 3, 1])

    N = 20000
    t = 1.0
    deltas = []

    for it in range(N):
        if it % 100 == 0:
            t += 10e-2

        alpha = LEARNING_RATE / t
        biggest_change = 0
        states_and_returns = play_game(grid, policy)
        seen_states = set()
        for s, G in states_and_returns:
            if s not in seen_states:
                old_theta = theta.copy()
                x = s2x(s)
                V_hat = theta.dot(x)

                theta += alpha * (G - V_hat) * x
                biggest_change = max(biggest_change, np.abs(old_theta - theta).sum())
                seen_states.add(s)
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
