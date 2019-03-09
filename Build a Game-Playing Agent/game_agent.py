"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    # My Original Heuristic adapated from improved_score in sample_players.py. Further fine tuning is done to accommodate the following:
    # Take max blank spaces which is the state when game starts
    # Take the blank spaces at the given point in time (m)
    # Take Legal moves available at given point in time (a,b)
    # max = game.width * game.height
    # m = len(game.get_blank_spaces())
    # a = len(game.get_legal_moves(player))
    # b = len(game.get_legal_moves(game.get_opponent(player)))
    # self_count = max - m - a
    # return float(self_count - 2 * opp_count)

    # Heuristic based on Suggestion given by reviewer
    max = game.move_count
    a = len(game.get_legal_moves(player))
    b = len(game.get_legal_moves(game.get_opponent(player)))
    self_count = a
    opp_count =  b
    w = 10/(max + 1)
    return float(self_count - w * opp_count)
    raise NotImplementedError


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    # My Original Heueristic
    # This is an adaptation of the improved_score heuristic in sample_players.py. Further fine tuning is done to accommodate the following:
    # Number of moves consumed during the game – game.move_count() (a)
    # Legal moves available at the given time for player and opponent (b,c)

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    # a = game.move_count
    # b = len(game.get_legal_moves(player))
    # c = len(game.get_legal_moves(game.get_opponent(player)))
    # # return float((len(game.get_blank_spaces()) - a) * b - c)
    # return float((b - a) - c)

    # Heuristic based on Suggestion given by reviewer
    score = .0
    max_space = game.width * game.height
    blank_space = len(game.get_blank_spaces())
    coefficient = float(max_space - blank_space) / float(max_space)

    a = game.get_legal_moves(player)
    b = game.get_legal_moves(game.get_opponent(player))

    for move in a:
        isNearWall = 1 if (move[0] == 0 or move[0] == game.width - 1 or
            move[1] == 0 or move[1] == game.height - 1) else 0
        score += 1 - coefficient * isNearWall

    for move in b:
        isNearWall = 1 if (move[0] == 0 or move[0] == game.width - 1 or
            move[1] == 0 or move[1] == game.height - 1) else 0
        score -= 1 - coefficient * isNearWall

    return score
    raise NotImplementedError


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    # This is an adaptation of the null score heuristic in sample_players.py.
    # Here we also assume no knowledge of terminal states and return a random number in the range (-10, 10).
    if game.is_loser(player):
       return float("-inf")

    if game.is_winner(player):
       return float("inf")

    return float(random.randrange(-10,10))
    raise NotImplementedError


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth, maximizing_player = True):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        # Adapted from minimax algorithm in the lecture tutorials
        # Initialize best score and best move
        best_score = float("-inf")
        best_move = None
        # loop through the avaialble legal moves from a max position using recursive functions
        # max value and min value helper functions defined
        # terminal test helper function defined
        # Depth is assumed to be one less from where the check starts
        # update best score and best move based on the result of the recursions
        for m in game.get_legal_moves():
            v = self.min_value(game.forecast_move(m),depth-1)
            if v > best_score:
                best_score = v
                best_move = m
        return best_move
        raise NotImplementedError

    def min_value(self,game,depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # Checks for terminal state first
        if self.terminal_test(game,depth):
            return self.score(game,self)
        # Assumes min value of v first and then does recursive max value Check
        # Depth is assumed to be one less from where the check starts
        v = float("inf")
        for m in game.get_legal_moves():
            v = min(v, self.max_value(game.forecast_move(m),depth-1))
        return v

    def max_value(self,game,depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # Checks for terminal state first
        if self.terminal_test(game,depth):
            return self.score(game,self)
        # Assumes max value of v first and then does recursive min value Check
        # Depth is assumed to be one less from where the check starts
        v = float("-inf")
        for m in game.get_legal_moves():
            v = max(v, self.min_value(game.forecast_move(m),depth-1))
        return v

    def terminal_test(self,game,depth):
        # Checks if there are any legal moves left or if depth still exists
        return not bool(game.get_legal_moves()) or depth <= 0

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        best_move = (-1, -1)
        # iterative deepening
        try:
            depth = 1
            while True:
                best_move = self.alphabeta(game,depth)
                depth += 1
        # TODO: finish this function!
        except SearchTimeout:
            pass
        return best_move
        raise NotImplementedError

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # TODO: finish this function!
        # Adapted from minimax algorithm done above
        # Initialize best score and best move
        best_score = float("-inf")
        best_move = None
        # loop through the avaialble legal moves from a max position using recursive functions
        # max value and min value helper functions defined
        # terminal test helper function defined
        # Depth is assumed to be one less from where the check starts
        # update best score and best move based on the result of the recursions
        # updates alpha based on the output
        for m in game.get_legal_moves():
            v = self.min_value(game.forecast_move(m),depth-1,alpha,beta)
            if v > best_score:
                best_score = v
                best_move = m
            alpha = max(alpha,best_score)
        return best_move
        raise NotImplementedError

    def min_value(self,game,depth,alpha,beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if self.terminal_test(game,depth):
            return self.score(game,self)
        # Assumes max value of v first and then does recursive max value Check
        # Depth is assumed to be one less from where the check starts
        # updates beta is output of function is less than beta
        v = float("inf")
        for m in game.get_legal_moves():
            v = min(v, self.max_value(game.forecast_move(m),depth-1,alpha,beta))
            beta = min(beta,v)
            # Checks for pruning
            if beta <= alpha:
                break
        return v

    def max_value(self,game,depth,alpha,beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if self.terminal_test(game,depth):
            return self.score(game,self)
        # Assumes min value of v first and then does recursive max value Check
        # Depth is assumed to be one less from where the check starts
        # updates alpha is output of function is more than alpha
        v = float("-inf")
        for m in game.get_legal_moves():
            v = max(v, self.min_value(game.forecast_move(m),depth-1,alpha,beta))
            alpha = max(alpha,v)
            # Checks for pruning
            if beta <= alpha:
                break
        return v

    def terminal_test(self,game,depth):
        # Checks if there are any legal moves left or if depth still exists
        return not bool(game.get_legal_moves()) or depth <= 0
        raise NotImplementedError
