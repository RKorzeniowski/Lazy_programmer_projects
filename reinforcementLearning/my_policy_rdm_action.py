import numpy as np
from my_grid_word import standard_grid, negative_grid
from iterative_policy import print_values, print_policy

# if agent is tring to go up it has 50% to succeed, and 0.5/3 to go left,right,back

# word is no longer determinisitc


SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')
prob_of_suc_act = 0.5


def roll_act(a):
    rdm_var = np.random.uniform()
    if rdm_var > prob_of_suc_act:
        pass
    elif rdm_var > prob_of_suc_act + prob_of_suc_act / 3:
        a = ('D')
    elif rdm_var > prob_of_suc_act + 2 * prob_of_suc_act / 3:
        a = ('L')
    else:
        a = ('R')
    return(a)


if __name__ == '__main__':
    # this grid gives you a reward of -0.1 for every non-terminal state
    # we want to see if this will encourage finding a shorter path to the goal
    # try it out with different penalty
    grid = negative_grid(step_cost=-1.0)

    print("rewards:")  # will print any dict with location as a key and reward as a value
    print_values(grid.rewards, grid)

    # initialize policy to random moves
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    # initialize random V[s]
    V = {}
    states = grid.all_states()
    for s in states:
        # other than terminal states initialize to random value
        if s in grid.actions:
            V[s] = np.random.random()
        else:
            # terminal state to 0
            V[s] = 0

    while True:

        # here we play with state transition probablity
        while True:
            biggest_change = 0
            for s in states:
                old_v = V[s]

                # we have to add state transition probability
                new_v = 0
                if s in policy:
                    for a in ALL_POSSIBLE_ACTIONS:
                        if a == policy[s]:
                            p = 0.5
                        else:
                            p = 0.5 / 3
                        grid.set_state(s)
                        r = grid.move(a)

                        new_v += p * (r + GAMMA * V[grid.current_state()])
                    V[s] = new_v

                    # break
                    biggest_change = max(biggest_change, np.abs(old_v - V[s]))

            if biggest_change < SMALL_ENOUGH:
                break

        # does it not happen to infiti when action is rdm?
        is_policy_conv = True
        # iterate through all the states (include terminal)
        for s in states:
            # we take states that are only in policy (no terminal states of wall)
            if s in policy:
                # save old state from possible states
                old_a = policy[s]
                # new state for later use
                new_a = None
                # initalize best val to always change
                best_value = float('-inf')
                # iterate through all possible actions = ('U', 'D', 'L', 'R')

                for a in ALL_POSSIBLE_ACTIONS:  # choose action
                    v = 0
                    for a2 in ALL_POSSIBLE_ACTIONS:  # resulting action
                        if a == a2:
                            p = 0.5
                        else:
                            p = 0.5 / 3
                        grid.set_state(s)
                        r = grid.move(a2)
                        v += p * (r + GAMMA * V[grid.current_state()])
                    if v > best_value:
                        best_value = v
                        # store action that gives best val
                        new_a = a
                # change policy to the action that gives best results
                policy[s] = new_a
                # if the action is not the same as before we refresh V function
                if new_a != old_a:
                    is_policy_conv = False

        if is_policy_conv:
            break

    print("valuse:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)
