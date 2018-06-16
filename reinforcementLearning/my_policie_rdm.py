import numpy as np

from my_grid_word import standard_grid, negative_grid
from iterative_policy import print_values, print_policy


# Step1 rmd initialize V(s) and pi(s)
SMALL_ENOUGH = 1e-3  # threshold for conergence
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

#this is deterministic
# all p(s',r|s,a) = 1 or 0

if __name__ == '__main__':

    # this frid fives you a reward of -0.1 for evert non termialn states
    # we want to see fi this will courage finding a shorteter path to the goal
    #gird = standard_grid()
    grid = negative_grid()

    # print rewards
    print("rewards:")  # will print any dict with location as a key and reward as a value
    print_values(grid.rewards, grid)

    # generate random policy like this
    # policy = {
    #     (2, 0): 'U',
    #     (1, 0): 'U',
    #     (0, 0): 'R',
    #     (0, 1): 'R',
    #     (0, 2): 'R',
    #     (1, 2): 'R',
    #     (2, 1): 'R',
    #     (2, 2): 'R',
    #     (2, 3): 'U',
    # }

    # state -> action
    # well randomly choose an actiona dn update as we learn
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    # inintial policy
    print("initial policy:")
    print_policy(policy, grid)

    # initialize random V[s]
    V = {}
    states = grid.all_states()
    for s in states:
        # V[s] = 0
        if s in grid.actions:
            V[s] = np.random.random()
        else:
            # terminal state
            V[s] = 0

    # Step2 iterative policy evaluation
    # repear until converges - will break out when policy does not change
    # This one is for deterministic version
    while True:

        while True:  # iterate until Value function does not improve anymore
            biggest_change = 0  # create var for later use
            for s in states:  # iterate tru all the possible states
                oldV = V[s]  # store state that we are now before we change it

                if s in policy:  # check if the state is in the policy
                    a = policy[s]  # what do we do when we are in this state
                    grid.set_state(s)  # move to the state that we are checking now
                    r = grid.move(a)  # check whats the reward when you move to the state thats connected with this one
                    V[s] = r + GAMMA * V[grid.current_state()]  # calculate the value of this state
                    biggest_change = max(biggest_change, oldV - V[s])  # check size of a chnage

            if SMALL_ENOUGH > biggest_change:
                break

        # Step3 Policy improvement
        # var to use later
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

                for a in ALL_POSSIBLE_ACTIONS:
                    # we set our agent to the state s that we are currently chekcing
                    grid.set_state(s)
                    # we perform action from our list of possible action from state s to get to state s'
                    # how do we know we are not doing sth illeagal (move to the wall or go out of the board (-1,0) )
                    # its checked if action is leagal inside move function
                    r = grid.move(a)  # if we give illegal move then it wont move and give back value of the same state that we are in
                    # standard bell eq deterministic without probability for each s' state
                    v = r + GAMMA * V[grid.current_state()]
                    # we check if previously stored val of v is better for our current action. If not we change best_value
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


print("valse:")
print_values(V, grid)
print("policy:")
print_policy(policy, grid)
