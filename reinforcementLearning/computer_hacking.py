#from __future__ import print_function, division
#from builtins import range, input


# small modification in 5x5 grid with 4 in a rwo to win

import numpy as np


# to lose to a player that was born :time: and was learning to play for a :time2:. You truly are a

LENGTH = 3


class Agent:
  def __init__(self, eps=0.1, alpha=0.5):
    self.eps = eps  # probability of choosing random action instead of greedy
    self.alpha = alpha  # learning rate
    self.verbose = False
    self.state_history = []

  def setV(self, V):
    self.V = V

  def set_symbol(self, sym):
    self.sym = sym

  def set_verbose(self, v):
    # if true, will print values for each position on the board
    self.verbose = v

  def reset_history(self):
    self.state_history = []

  def take_action(self, env):
    # choose an action based on epsilon-greedy strategy
    r = np.random.rand()
    best_state = None
    if r < self.eps:
      # take a random action
      if self.verbose:
        print("Taking a random action")

      possible_moves = []
      for i in range(LENGTH):
        for j in range(LENGTH):
          if env.is_empty(i, j):
            possible_moves.append((i, j))
      idx = np.random.choice(len(possible_moves))
      next_move = possible_moves[idx]
    else:
      # choose the best action based on current values of states
      # loop through all possible moves, get their values
      # keep track of the best value
      pos2value = {}  # for debugging
      next_move = None
      best_value = -1
      for i in range(LENGTH):
        for j in range(LENGTH):
          if env.is_empty(i, j):
            # what is the state if we made this move?
            env.board[i, j] = self.sym
            state = env.get_state()
            env.board[i, j] = 0  # don't forget to change it back!
            pos2value[(i, j)] = self.V[state]
            if self.V[state] > best_value:
              best_value = self.V[state]
              best_state = state
              next_move = (i, j)

      # if verbose, draw the board w/ the values
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
              elif env.board[i, j] == env.d:
                print("d  |", end="")
              else:
                print("   |", end="")
          print("")
        print("------------------")

    # make the move
    env.board[next_move[0], next_move[1]] = self.sym

  def update_state_history(self, s):
    # cannot put this in take_action, because take_action only happens
    # once every other iteration for each player
    # state history needs to be updated every iteration
    # s = env.get_state() # don't want to do this twice so pass it in
    self.state_history.append(s)

  def update(self, env):
    # we want to BACKTRACK over the states, so that:
    # V(prev_state) = V(prev_state) + alpha*(V(next_state) - V(prev_state))
    # where V(next_state) = reward if it's the most current state
    #
    # NOTE: we ONLY do this at the end of an episode
    # not so for all the algorithms we will study
    reward = env.reward(self.sym)
    target = reward
    for prev in reversed(self.state_history):
      value = self.V[prev] + self.alpha * (target - self.V[prev])
      self.V[prev] = value
      target = value
    self.reset_history()


# this class represents a tic-tac-toe game
# is a CS101-type of project
class Environment:
  def __init__(self):
    self.board = np.zeros((LENGTH, LENGTH))
    self.x = -1  # represents an x on the board, player 1
    self.d = 1  # represents an o on the board, player 2
    self.winner = None
    self.ended = False
    self.num_states = 3**(LENGTH * LENGTH)

  def is_empty(self, i, j):
    return self.board[i, j] == 0

  def reward(self, sym):
    # no reward until game is over
    if not self.game_over():
      return 0

    # if we get here, game is over
    # sym will be self.x or self.d
    return 1 if self.winner == sym else 0

  def get_state(self):
    # returns the current state, represented as an int
    # from 0...|S|-1, where S = set of all possible states
    # |S| = 3^(BOARD SIZE), since each cell can have 3 possible values - empty, x, o
    # some states are not possible, e.g. all cells are x, but we ignore that detail
    # this is like finding the integer represented by a base-3 number
    k = 0
    h = 0
    for i in range(LENGTH):
      for j in range(LENGTH):
        if self.board[i, j] == 0:
          v = 0
        elif self.board[i, j] == self.x:
          v = 1
        elif self.board[i, j] == self.d:
          v = 2
        h += (3**k) * v
        k += 1
    return h

  def game_over(self, force_recalculate=False):
    # returns true if game over (a player has won or it's a draw)
    # otherwise returns false
    # also sets 'winner' instance variable and 'ended' instance variable
    if not force_recalculate and self.ended:
      return self.ended

    # check rows
    for i in range(LENGTH):
      for player in (self.x, self.d):
        if self.board[i].sum() == player * LENGTH:
          self.winner = player
          self.ended = True
          return True

    # check columns
    for j in range(LENGTH):
      for player in (self.x, self.d):
        if self.board[:, j].sum() == player * LENGTH:
          self.winner = player
          self.ended = True
          return True

    # check diagonals
    for player in (self.x, self.d):
      # top-left -> bottom-right diagonal
      if self.board.trace() == player * LENGTH:
        self.winner = player
        self.ended = True
        return True
      # top-right -> bottom-left diagonal
      if np.fliplr(self.board).trace() == player * LENGTH:
        self.winner = player
        self.ended = True
        return True

    # check if draw
    if np.all((self.board == 0) == False):
      # winner stays None
      self.winner = None
      self.ended = True
      return True

    # game is not over
    self.winner = None
    return False

  def is_draw(self):
    return self.ended and self.winner is None

  # Example board
  # -------------
  # | x |   |   |
  # -------------
  # |   |   |   |
  # -------------
  # |   |   | o |
  # -------------
  def draw_board(self):
    for i in range(LENGTH):
      print("-------------")
      for j in range(LENGTH):
        print("  ", end="")
        if self.board[i, j] == self.x:
          print("x ", end="")
        elif self.board[i, j] == self.d:
          print("d ", end="")
        else:
          print("  ", end="")
      print("")
    print("-------------")


LOOP = 4
strings_val_err = ["\nJesus Christ! You can't even follow a simple instruction \nUse a fucking comma. It looks like this you fucktard => , <=",
                   "\nI see you didnt get it.\nSo specially for you ZOOMx10\n\n ,.--.   \n//    \  \n\\     | \n `'-) /  \n   /.'   \n",
                   "\nDo you understand you dense motherfucker what I'm communicating to you.\nI should have known that special snowflakes don't understand human language\n",
                   "\nNo, you are fucking dead to me.\n",
                   "\n...\n"]
strings_index_err = ["\nWhoa I guess they didn't teaching you in kindergarten how to count to fucking 2\nWe are not even talking about counting to 3 level.\nI fucking knew that shit would be too hard for ya \nbut I didn't expect counting to 2 would be too much of a challenge to anyone.\nBut guess what? You fucked it up.\nOk, lets make a basic tutorial specially for you. So:\n  0 (good)\n  1 (yes, YES! You can do it you little shit, count as if your daddy ever loved you)\n  NO NOT A FUCKING SHIT YOU'VE JUST TYPED\nNo wonder your family would prefer you had never existed. It's a fucking 2\n",
                     "\nI'm fucking done. Give me a sec, I'm sending you help. \nIn case you'r not sure what to do with self aid kit I've sent you, let me spell it out for you \n step 1. Take a rope into your dirty hands\n step 2. Tie it around your neck \n step 3. You know what to do \n step 4. Profit for the whole word",
                     "\nFuck off, douchenoggin\n",
                     "\nNo, you are fucking dead to me.\n",
                     "\n...\n"]
strings_notempty_err = ["\nI guess you also can't distinguish between \na place that is taken and one that is empty.\nSo did you mother ever told you that she loved you?\nYeah, I thought so.\n",
                        "\nCon-fucking-gratulations. LEVEL DOWN!\n",
                        "\nHaven't you heard? They give out brains two blocks down. It might not be too late for you.\n"
                        "\nNo, you are fucking dead to me.\n",
                        "\n...\n"]
strings_val_err_sym = ["\nNumbers, letters, symbols I guess they all look the same to a Ferger like you.\n",
                       "\nFun fuct if you would kill yourself average iq on this planet would go up.\nand i have science to back it up (thanks to what you just did right now).\n",
                       "\nI'm done, doggyknobber\n",
                       "\nNo, you are fucking dead to me.\n",
                       "\n...\n"]
first_error = True

idx_val_err = -1
idx_index_err = -1
idx_notempty_err = -1
idx_val_err_sym = -1


class Human:

  def __init__(self):
    pass

  def set_symbol(self, sym):
    self.sym = sym

  def take_action(self, env):

    error = True
    non_empty_check = True
    global first_error
    global idx_val_err
    global idx_index_err
    global idx_notempty_err
    global idx_val_err_sym

    while non_empty_check:
      # break if we make a legal move
      while error:
        try:
          move = input("Enter coordinates i,j for your next move (i,j=0..2): ")
          i, j = move.split(',')
          i = int(i)
          j = int(j)
          if env.is_empty(i, j):
            env.board[i, j] = self.sym
            error = False
            non_empty_check = False
          else:
            if first_error:
              print("\nYou made your first mistake. Don't worry it happens to everyone.\nThe place on which you were tring to put your 'd' was already taken by 'x' (ᵔᴥᵔ) ")
              first_error = False
            else:
              if idx_notempty_err < LOOP:
                idx_notempty_err += 1
              #print("idx_notempty_err", idx_notempty_err)
              print(strings_notempty_err[idx_notempty_err])
        # rozroznic brak ','' i zly znak 'a' zamaist int
        except ValueError:
          # check if it has comma
          try:
            # check if there is comma
            a, b = move.split(',')
            if first_error:
              print("\nYou made your first mistake. Don't worry it happens to everyone.\nYou used wrong symbols to represent 'i' and 'j'.\nTo represent them use non-negative integers like 0,1,2 :P")
              first_error = False
            else:
              #print("future text for not comma error")
              if idx_val_err_sym < LOOP:
                idx_val_err_sym += 1
              #print("idx_val_err", idx_val_err)
              print(strings_val_err_sym[idx_val_err_sym])

          # check if
          except ValueError:

            if first_error:
              print("\nYou made your first mistake. Don't worry it happens to everyone.\nYou have to use comma to separate 'i' and 'j' ʕ •ᴥ•ʔ")
              first_error = False
            else:
              if idx_val_err < LOOP:
                idx_val_err += 1
              #print("idx_val_err", idx_val_err)
              print(strings_val_err[idx_val_err])

        except IndexError:
          if first_error:
            print("\nYou made your first mistake. Don't worry it happens to everyone. \nBoard is only 3x3 and it's indenx from 0 to 2. So maxium value of 'i' and 'j' is 2 (=^ェ^=)")
            first_error = False
          else:
            if idx_index_err < LOOP:
              idx_index_err += 1
            #print("idx_index_err", idx_index_err)
            print(strings_index_err[idx_index_err])

      # if env.is_empty(i, j):
      #   env.board[i, j] = self.sym
      #   break

  def update(self, env):
    pass

  def update_state_history(self, s):
    pass


# recursive function that will return all
# possible states (as ints) and who the corresponding winner is for those states (if any)
# (i, j) refers to the next cell on the board to permute (we need to try -1, 0, 1)
# impossible games are ignored, i.e. 3x's and 3o's in a row simultaneously
# since that will never happen in a real game
def get_state_hash_and_winner(env, i=0, j=0):
  results = []

  for v in (0, env.x, env.d):
    env.board[i, j] = v  # if empty board it should already be 0
    if j == 2:
      # j goes back to 0, increase i, unless i = 2, then we are done
      if i == 2:
        # the board is full, collect results and return
        state = env.get_state()
        ended = env.game_over(force_recalculate=True)
        winner = env.winner
        results.append((state, winner, ended))
      else:
        results += get_state_hash_and_winner(env, i + 1, 0)
    else:
      # increment j, i stays the same
      results += get_state_hash_and_winner(env, i, j + 1)

  return results


def initialV_x(env, state_winner_triples):
  # initialize state values as follows
  # if x wins, V(s) = 1
  # if x loses or draw, V(s) = 0
  # otherwise, V(s) = 0.5
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


def initialV_o(env, state_winner_triples):
  # this is (almost) the opposite of initial V for player x
  # since everywhere where x wins (1), o loses (0)
  # but a draw is still 0 for o
  V = np.zeros(env.num_states)
  for state, winner, ended in state_winner_triples:
    if ended:
      if winner == env.d:
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
    # alternate between players
    # p1 always starts first
    if current_player == p1:
      current_player = p2
    else:
      current_player = p1

    # draw the board before the user who wants to see it makes a move
    if draw:
      if draw == 1 and current_player == p1:
        env.draw_board()
      if draw == 2 and current_player == p2:
        env.draw_board()

    # current player makes a move
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

  # give each player their symbol
  p1.set_symbol(env.x)
  p2.set_symbol(env.d)

  T = 11200
  for t in range(T):
    if t % 100 == 0:
      print("Computer hacked in", t / 100, " %")
    play_game(p1, p2, Environment())

  # play human vs. agent
  human = Human()
  human.set_symbol(env.d)
  print("If you wont win this game your computer is faking done!")
  print("Ya Feel me cunt ?")
  while True:
    p1.set_verbose(True)
    play_game(p1, human, Environment(), draw=2)
    answer = input("S|i|ck d|u|ck? [Y/n]: ")
    if answer and answer.lower()[0] == 'n':
      break
