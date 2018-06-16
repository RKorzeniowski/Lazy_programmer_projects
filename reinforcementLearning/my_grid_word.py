import numpy as np
import matplotlib.pyplot as plt


# class Agent:
#     pass


class Grid:  # Enviroment
    def __init__(self, width, height, start):
        self.width = width
        self.height = height
        self.i = start[0]
        self.j = start[1]

    def set(self, rewards, actions):
        # reward should be a dict of: (i, j): r(row,col): reward
        # actions should be a dict of: (i, j): A(row, col): list of possible actions
        self.rewards = rewards
        self.actions = actions

    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        return (self.i, self.j)

    def is_terminal(self, s):
        return s not in self.actions  # check if there are any actions that can be made from this state. If there are non returns True

    def move(self, action):
        # check if legal move first
        if action in self.actions[(self.i, self.j)]:  # chck if action is in the action dictionary
            if action == 'U':
                self.i -= 1
            if action == 'D':
                self.i += 1
            if action == 'L':
                self.j -= 1
            if action == 'R':
                self.j += 1
        # return a reward (if any)
        return self.rewards.get((self.i, self.j), 0)  # reward from action dict

    def undo_move(self, action):
        # these are the opposite of what U/D/L/R should normally do
        if action == 'U':
            self.i += 1
        if action == 'D':
            self.i -= 1
        if action == 'L':
            self.j += 1
        if action == 'R':
            self.j -= 1
        # raise an exception if we arrive somwhere we shoudnt be
        # should never happen
        assert(self.current_state() in self.all_states())

    def game_over(self):
        # returns true if game is over else fasle
        # true if we are in the a state where no actions are possible
        return (self.i, self.j) not in self.actions

    def all_states(self):
        # all possibly buddy but simple wat to get all states
        # either a postions taht has possible nexy actions
        # or a positon that yields a reward
        return set(self.actions.keys() | self.rewards.keys())


def standard_grid():
    #s - start
    # x - cant go there
    # . . . 1
    # . x .-1
    # s . . .
    g = Grid(3, 4, (2, 0))
    rewards = {(0, 3): 1, (1, 3): -1}
    actions = {
        (0, 0): ('D', 'R'),
        (0, 1): ('L', 'R'),
        (0, 2): ('L', 'D', 'R'),
        (1, 0): ('U', 'D'),
        (1, 2): ('U', 'D', 'R'),
        (2, 0): ('U', 'R'),
        (2, 1): ('L', 'R'),
        (2, 2): ('L', 'R', 'U'),
        (2, 3): ('L', 'U'),
    }

    g.set(rewards, actions)
    return g


def negative_grid(step_cost=-0.1):
    g = standard_grid()
    g.rewards.update({
        (0, 0): step_cost,
        (0, 1): step_cost,
        (0, 2): step_cost,
        (1, 0): step_cost,
        (1, 2): step_cost,
        (2, 0): step_cost,
        (2, 1): step_cost,
        (2, 2): step_cost,
        (2, 3): step_cost,
    })
    return g


def play_game(agent, env):
    pass


if __name__ == '__main__':
    g = negative_grid()

    for i in g.rewards:
        print("rewards", i)

    print("\n")

    for i in g.actions:
        print("actions", i)

    for i in g.actions:
        print("actions", g.actions[i])

    for i in g.actions:
        print("rewards for all avaliable actions", g.rewards[i])

    for i in g.all_states():
        print("rewards for all states", g.rewards[i])

    for i in g.all_states():
        if i in g.actions:
            print("all states/action filter", i)

    for i in g.all_states():
        print("all states", i)