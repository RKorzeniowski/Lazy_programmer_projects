import numpy as np
import matplotlib.pyplot as plt
from iterative_policy import print_values, print_policy
from my_grid_word import standard_grid, negative_grid

GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')
# change from action A to other 3


def random_action(a, eps=0.1):
    p = np.random.random()
    if p < (1 - eps):
        return a
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)


def play_game(grid, policy):
    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])

    s = (2, 0)
    grid.set_state(s)
    a = random_action(policy[s])
    # each tiple is s(t), a(t), r(t)
    # but r(t) results from taking action a(t-1) from s(t-1) and landing in state s(t)
    states_actions_rewards = [(s, a, 0)]

    while True:
        r = grid.move(a)
        s = grid.current_state()
        if grid.game_over():
            states_actions_rewards.append((s, None, r))
            break
        else:
            a = random_action(policy[s])
            states_actions_rewards.append((s, a, r))

    # G = 0
    # states_actions_returns = []
    # first = True
    # for s, a, r in reversed(states_actions_rewards):
    #     if first:
    #         first = False
    #     else:
    #         states_actions_returns.append((s, a, G))
    #     G = r + GAMMA * G
    # states_actions_returns.reverse()
    # return states_actions_returns

    G = 0
    states_actions_returns = []
    first = True
    for s, a, r in reversed(states_actions_rewards):
        if first:
            first = False
        else:
            states_actions_returns.append((s, a, G))
        G = r + GAMMA * G
    states_actions_returns.reverse()  # we want it to be in order of state visited
    return states_actions_returns


def max_dict(d):
    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val


if __name__ == '__main__':
    grid = negative_grid(step_cost=-0.1)

  # print rewards
    print("rewards:")
    print_values(grid.rewards, grid)

    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    Q = {}
    returns = {}
    states = grid.all_states()
    # loop through all the states include terminal
    for s in states:
        # initialize all the state values to 0
        if s in grid.actions:
            Q[s] = {}
            for a in ALL_POSSIBLE_ACTIONS:
                Q[s][a] = 0
                returns[(s, a)] = []
        else:
            pass

    N = 5000
    deltas = []

    for t in range(N):
        if t % 1000 == 0:
            print(t)

        biggest_change = 0
        states_actions_returns = play_game(grid, policy)
        seen_state_action_pairs = set()
        for s, a, G in states_actions_returns:
            sa = (s, a)
            if sa not in seen_state_action_pairs:
                old_q = Q[s][a]
                returns[sa].append(G)
                Q[s][a] = np.mean(returns[sa])
                biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
                seen_state_action_pairs.add(sa)
        deltas.append(biggest_change)

        # update policy
        for s in policy.keys():
            policy[s] = max_dict(Q[s])[0]

    plt.plot(deltas)
    plt.show()

    V = {}
    for s, Qs in Q.items():
        V[s] = max_dict(Q[s])[1]

    print("valuse:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)
