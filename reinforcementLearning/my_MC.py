from __future__ import print_function, division

import numpy as np
from iterative_policy import print_values, print_policy
from my_grid_word import standard_grid, negative_grid

SMALL_ENOUGH = 1e-3  # threshold for conergence
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


def play_game(grid, policy):
    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])

    s = grid.current_state()
    states_and_rewards = [(s, 0)]
    while not grid.game_over():
        a = policy[s]
        r = grid.move(a)
        s = grid.current_state()
        states_and_rewards.append((s, r))

    G = 0
    states_and_returns = []
    first = True
    for s, r in reversed(states_and_rewards):
        if first:
            first = False
        else:
            states_and_returns.append((s, G))
        G = r + GAMMA * G
    states_and_returns.reverse()
    return states_and_returns


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
        (1, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U',
    }

    states = grid.all_states()
    V = {}
    returns = {}

    N = 300
    # loop through all the states include terminal
    for s in states:
        # initialize all the state values to 0
        if s in grid.actions:
            returns[s] = []
        else:
            V[s] = 0

    for t in range(N):
        states_and_returns = play_game(grid, policy)
        seen_states = set()
        for s, G in states_and_returns:
            if s not in seen_states:
                returns[s].append(G)
                V[s] = np.mean(returns[s])
                seen_states.add(s)

    print("valse:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)


# def first_visit_monte(pi, N):
#     gird = standard_grid()


#     all_returns = {}

#     for i in range(N):
#         states, returns = play_episode
#         for s, g in zip(states, returns):
#             check = s.append()
#             if s not in set(check):
#                 all_returns[s].append(g)
#                 V[s] = sample_mean(all_returns[s])
#     return V

# # init policy


# # calculate returns from rewards
# s = grid.current_state()
# states_and_returns = [(s, 0)]
# while not game_over:
#     a = policy[s]
#     r = grid.move(a)
#     s = grid.current_state
#     states_and_returns.append((s, r))

#     G = 0
#     states_and_returns = []
#     for s, r in reverse(states_and_returns):
#         states_and_returns.append((s, G))
#         G = r + gamma * G
#     states_and_returns.reverse()
