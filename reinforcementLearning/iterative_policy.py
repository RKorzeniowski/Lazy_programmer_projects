from __future__ import print_function, division

import numpy as np

from my_grid_word import standard_grid, negative_grid


SMALL_ENOUGH = 1e-3  # threshold for conergence


def print_values(V, g):
    for i in range(g.width):
        print("---------------------------")
        for j in range(g.height):
            v = V.get((i, j), 0)
            if v >= 0:
                print(" %.2f|" % v, end="")
            else:
                print("%.2f|" % v, end="")  # -ve sign takes up an extra space
        print("")


def print_policy(P, g):
    for i in range(g.width):
        print("---------------------------")
        for j in range(g.height):
            a = P.get((i, j), ' ')
            print("  %s  |" % a, end="")
        print("")


if __name__ == '__main__':
    # create a grid
    grid = standard_grid()

    # get all the possible states(terminal but not wall).
    states = grid.all_states()

    # empty dictionary for values
    V = {} 

    # loop through all the states include terminal
    for s in states:
        # initialize all the state values to 0
        V[s] = 0
        # how value changes with time
        gamma = 1.0 
        
    # loop until value of V at any point does not change
    while True:
        # var for change in V
        biggest_change = 0
        # iterate through all the states
        for s in states:
            # save current state for later use
            old_v = V[s]

            # take only states that are in actions dict (no terminal states)
            if s in grid.actions:
                # var for later use
                new_v = 0
                # p_a is a probability of performing certain action 
                # given the number of all the possible actions that can be performent in this state = len(grid.actions[s])
                p_a = 1.0 / len(grid.actions[s])
                # now we loop through all actions that can be made from state s
                for a in grid.actions[s]:
                    # we set our agent to the state s that we are currently chekcing
                    grid.set_state(s)
                    # we perform action from our list of possible action from state s to get to state s'
                    r = grid.move(a)
                    # thanks to moving to state s' we now know reward r and Value of this state (V[grid.current_state())
                    # we multiply it by probability of this action
                    # then acumulate value of state s to V[s] for every action a 
                    new_v += p_a * (r + gamma * V[grid.current_state()])
                # save new state
                V[s] = new_v
                # check if new state is different from original.
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))
        # if new state if close enought to old state we break from the loop
        if biggest_change < SMALL_ENOUGH:
            break
    print('vales for uniformy random actions:')
    print_values(V, grid)
    print('\n\n')

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
    print_policy(policy, grid)

    V = {}
    for s in states:
        V[s] = 0

    gamma = 0.9

    while True:
        biggest_change = 0
        for s in states:
            old_v = V[s]

            if s in policy:
                a = policy[s]
                grid.set_state(s)
                r = grid.move(a)
                V[s] = r + gamma * V[grid.current_state()]
                print(V[grid.current_state()])
                # break
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))

        if biggest_change < SMALL_ENOUGH:
            break
    print("values for fixed policiy:")
    print_values(V, grid)

    #grid = negative_grid()

    # for s in states:
    #     # print(s)
    #     if s in policy:
    #         print(policy)
    # for s in grid.actions.keys():
    #    V[grid.current_state()]
    #
    states = grid.all_states()
    print(states)
