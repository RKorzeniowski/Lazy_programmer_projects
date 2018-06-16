import numpy as np


LENGHT = 3


class Environment:
    def __init__(self):
        self.board = np.zeros((LENGHT, LENGHT))
        self.x = -1  # reprezents x on the board - player 1
        self.o = 1  # reprezents o on the board - player 2
        self.winner = None
        self.ended = False
        self.num_states = 3**(LENGHT * LENGHT)

    # retrun true if (i,j) element is empty
    def is_empty(self, i, j):
        return self.board[i, j] == 0

    # sym = symbol of a player (x or o)
    def reward(self, sym):
        # no reward ntil game is over
        if not self.game_over():
            return 0
        # if we get here, game is over
        # sym will be slef.x or self.o
        return 1 if self.winner == sym else 0

    def get_state(self):
        k = 0
        h = 0

        for i in range(LENGHT):
            for j in range(LENGHT):
                if self.board[i, j] == 0:
                    v = 0
                elif self.board[i, j] == self.x:
                    v = 1
                elif self.board[i, j] == self.o:
                    v = 2
                h += (3**k) * v
                k += 1
        return h

    def game_over(self, force_recalculate=False):
        if not force_recalculate and self.ended:
            return self.ended

        # check rows
        for i in range(LENGHT):
            for player in (self.x, self.o):
                if self.board[i].sum() == player * LENGHT:
                    self.winner = player
                    self.ended = True
                    return True

        # check columns
        for j in range(LENGHT):
            for player in (self.x, self.o):
                if self.board[:, j].sum() == player * LENGHT:
                    self.winner = player
                    self.ended = True
                    return True

        # what about diagonals
        for player in (self.x, self.o):
            # top-left 2 bottom right diagonal (trace of a matrix)
            if self.board.trace() == player * LENGHT:
                self.winner = player
                self.ended = True
                return True

            if np.fliplr(self.board).trace() == player * LENGHT:
                self.winner = player
                self.ended = True
                return True

        # check if draw (to dziala chyba tylko dla nieparzystej liczby wymiarow)
        if np.all((self.board == 0) == False):
            # winner stays None
            self.winner = None
            self.ended = True
            return True

        #game is not over
        self.winner = None
        return False

    def draw_board(self):
        for i in range(LENGTH):
            print("-------------")
            for j in range(LENGTH):
                print("  ", end="")
                if self.board[i, j] == self.x:
                    print("x ", end="")
                elif self.board[i, j] == self.o:
                    print("o ", end="")
                else:
                    print("  ", end="")
            print("")
        print("-------------")


class Agent():
    def __init__(self, eps=0.1, alpha=0.5):
        self.eps = eps  # prob of choosing rdm action instead of greedy
        self.alpha = alpha  # learning rate
        self.verbose = False
        self.state_history = []

    def setV(self, V):
        self.V = V

    def set_symbol(self, symbol):
        self.symbol = symbol

    def set_verbose(self, v):
        self.verbose = v

    def restart_history(self):
        self.state_history = []

    def take_action(self, env):
        r = np.random.rand()
        best_state = None
        if r < self.eps:
            # take a random action
            if self.verbose:
                print("taking a random action")

            possible_moves = []
            for i in range(LENGHT):
                for j in range(LENGHT):
                    if env.is_empty(i, j):
                        possible_moves.append((i, j))
            ifx = np.random.choice(len(possible_moves))
            next_move = possible_moves[idx]
        # else:
        #     next_move = None
        #     best_value = -1
        #     for i in range(LENGHT)
        #         for j in range(LENGHT):
        #             if env.is_empty(i, j):
        #                 # what is the state if we made this move?
        #                 env.board[i, j] = self.sym
        #                 state = env.get_state()
        #                 env.board[i, j] = 0  # dont forget o change it back
        #                 if self.V[state] > best_value:
        #                     best_value = self.V[state]
        #                     best_state = statenext_move = (i, j)

        else:
            pos2vale = {}  # for debugging
            next_move = None
            best_value = -1
            for i in range(LENGHT):
                for j in range(LENGHT):
                    if env.is_empty(i, j):
                        # what is the state if we made this ove?
                        env.board[i, j] = self.sym
                        state = env.get_state()
                        env_board[i, j] = 0
                        pos2value[(i, j)] = self.V[state]
                        if self.V[state] > best_value:
                            best_value = self.V[state]
                            best_state = state
                            next_move = (i, j)

            if self.verbose:
                print("Taking a greedy action")
                for i in range(LENGTH):
                    print("------------------")
                    for j in range(LENGTH):
                        if env.is_empty(i, j):
                            # print the value
                            print(" %.2f|" % pos2value[(i, j)], end="")
                        else:
                            print("  ", end="")
                            if env.board[i, j] == env.x:
                                print("x  |", end="")
                            elif env.board[i, j] == env.o:
                                print("o  |", end="")
                            else:
                                print("   |", end="")
                    print("")
                print("------------------")

    def update_state_history(self, s):
        self.state_history.append(s)

    def update(env):
        reward = env.reward(self.sym)
        target = reward
        for prev in reversed(self.state_history):
            value = self.V[prev] + self.alpha * (target - self.V[prev])
            self.V[prev] = value
            target = value
        self.restart_history()

    # could be done with dict
    #        np.array([0,0,0],
    # board ->          [0,0,0],
    #                 [0,0,0])

    #        np.array([0,0,0],
    # board ->          [0,1,0],
    #                 [0,0,0])

    #        np.array([2,0,0],
    # board ->          [0,1,0], state -> 200010000
    #                 [0,0,0])

    # satate
    # board

    # Vs = 1 if player wins
    # Vs = 0 if lose or draw
    # Vs = 0.5 all else
    # generate all the states by perumations

    # example of permutation generation ()
    # def generate_all_binary_numbers(N):
    #     results = []
    #     if N>0:
    #       child_results = generate_all_binary_numbers(N-1)
    #     for prefix in ('0', '1'):
    #         for suffix in child_results:
    #             new_result = prefix + suffix
    #             results.append(new_result)
    #     return results


def get_state_hash_and_winner(env, i=0, j=0):
    results = []

    for v in (0, env.x, env.o):
        env.board[i, j] = v  # if empty boardit should already be 0
        if j == 2:
            # j goes back to 0, increase i, unless i = 2 then we are done
            if i == 2:
                # board is full , collect results and return
                state = env.get_state()
                ended = env.game_over(force_recalculate=True)
                winner = env.winner
                results.append((state, winner, ended))
            else:
                results += get_state_hash_and_winner(env, i + 1, 0)
        else:
            # j goes up, i the same
            results += get_state_hash_and_winner(env, i, j + 1)

    return results


def initialV_x(env, state_winnder_triples):
    # initilaize state values as folllows
    # if x wins , V(s)+ 1
    # if x loses or draw V(s)= 0
    # otherwise V(s) = 0.5
    V = np.zeros(env.num_states)
    for state, winner, ended in state_winner_triples:
        if ended:
            if winner == env.x:
                v = 1
            else:
                v = 0
        else:
            v = 0.5
        V[state] = v
    return V


def initialV_o(env, state_winnder_triples):
    # initilaize state values as folllows
    # if x wins , V(s)+ 1
    # if x loses or draw V(s)= 0
    # otherwise V(s) = 0.5
    V = np.zeros(env.num_states)
    for state, winner, ended in state_winner_triples:
        if ended:
            if winner == env.o:
                v = 1
            else:
                v = 0
        else:
            v = 0.5
        V[state] = v
    return V


def play_game(p1, p2, env, draw=False):

    # loops until the game is over
    current_player = None
    while not env.game_over():
        # alternate between platers
        # p1 always starts first
        if current_player == p1:
            current_player == p2
        else:
            current_player == p1

        # draw the board begore the user who wants to see to make a move
        if draw:
            if draw == 1 and current_player == p1:
                env.draw_board()
            if draw == 2 and current_player == p2:
                env.draw_board()

        # current player make a move
        current_player.take_action(env)

        # update state histories
        state = env.get_state()
        p1.update_state_history(state)
        p2.update_state_history(state)

    if draw:
        env.draw_board()

    # do the value function update
    p1.update(env)
    p2.update(env)


class Human(object):
    """docstring for Human"""

    def __init__(self, arg):
        pass

    def set_symbol(self, sym):
        self.sym = sym

    def take_action(self, env):
        while True:
            # brake if we make a legal move
            move = raw_input("enter coordinates i,j for your next move (i,j=0..2)")
            i, j = move.split(',')
            i = int(i)
            j = int(j)
            if env.is_empty(i, j):
                env_board[i, j] = self.sym
                break

    def update(self, env):
        pass

    def update_state_history(self, s):
        pass


if __name__ == '__main__':
    # train the agent
    p1 = Agent()
    p2 = Agent()

    # set initial V for p1 and p2
    env = Environment()
    state_winner_triples = get_state_hash_and_winner(env)

    Vx = initialV_x(env, state_winner_triples)
    p1.setV(Vx)
    Vo = initialV_o(env, state_winner_triples)
    p2.setV(Vo)

    # give each player thier symbol
    p1.set_symbol(env.x)
    p2.set_symbol(env.o)

    # train AI
    T = 10000
    for t in range(T):
        if t % 200 == 0:
            print(t)
        play_game(p1, p2, Environment())

    # play human vs. agent
    human = Human()
    human.set_symbol(env.o)
    while True:
        p1.set_verbose(True)
        play_game(p1, human, Environment(), draw=2)
        # AI as p1 to check if he takes center
        anwser = raw_input('Play again? [Y/n]: ')
        if anwser and anwser.lower()[0] == 'n':
            break
