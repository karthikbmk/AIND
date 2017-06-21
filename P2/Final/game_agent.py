"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def penalize_and_reward(game, player,reward_coffiecient=2,penalty_coefficient=2):
	"""
	Reward the player with reward_coffiecient for the number of moves he has.
	Penalize the opponent player with penalty_coefficient for the number of moves he has.
	"""
	if game.is_winner(player):
		return float("inf")
	elif game.is_loser(player):
		return float("-inf")
	else:
		my_move_count = len(game.get_legal_moves(player))
		opp_move_count = len(game.get_legal_moves(game.get_opponent(player)))
		
		return float(abs(reward_coffiecient*my_move_count - penalty_coefficient*opp_move_count))

def custom_score_1(game, player):
	"""
	High Reward for the number of moves the player has, and low penalty 
	for the number of moves opponent has
	"""
	return penalize_and_reward(game,player,5,2)
	
def custom_score_2(game, player):
	"""
	Low Reward for the number of moves the player has, and High penalty 
	for the number of moves opponent has
	"""
	return penalize_and_reward(game,player,2,4)	
	
def custom_score_3(game, player):
	"""
	1 Reward for the number of moves the player has, and 1 penalty 
	for the number of moves opponent . (Equal rewards and penalty)
	"""
	return penalize_and_reward(game,player,1,1)	

def custom_score(game, player):
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
    return custom_score_2(game, player)


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
        self.maximizing_player = True

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
        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        
        moves = game.get_legal_moves()
        
        if not moves:
            return (-1,-1)
        else:
            best_move = moves[random.randint(0, len(moves) - 1)]
        
        try:
            return self.minimax(game,self.search_depth)
        
        except SearchTimeout:
            return best_move
    
    def minimax (self, game, depth):
        score,move = self.minimax_2(game, depth,True)
        return move
    def minimax_2(self, game, depth,maximizing_player=True):
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
		
        moves = game.get_legal_moves()
        
        if (depth == 0 or len(moves) == 0):                                
            return self.score(game,self),(-1,-1) 
        
        
        util_queue = []
        
        max_score = float("-inf")
        min_score = float("inf")
        
        max_move = (-1,-1)
        min_move = (-1,-1)
        
        for move in moves:                                 
            new_game = game.forecast_move(move)                          
            maximizing_player = not maximizing_player
            score,_move = self.minimax_2(new_game, depth-1,maximizing_player)  
            if score > max_score:
                max_score = score
                max_move = move
            if score < min_score:
                min_score = score 
                min_move = move
                
            #util_queue.append(score)

        if (self.maximizing_player):
            #best_score = max(util_queue)
            best_score = max_score
            best_move = max_move
        else:
            #best_score = min(util_queue)
            best_score = min_score
            best_move = min_move
            
        #best_move = moves[util_queue.index(best_score)]
        
        return best_score, best_move


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
        best_move = (-1,-1)
        depth = 1
        
        try:
            while(1==1):
                best_move = self.alphabeta(game,depth)
                depth += 1
        
        except SearchTimeout:
            return best_move

        # TODO: finish this function!
        #raise NotImplementedError

    def max_val(self, game, depth, alpha, beta,maximizing_player):
	
	
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()	
            
        moves = game.get_legal_moves()
		
        if (depth == 0 or len(moves) == 0):                                
            return self.score(game,self),(-1,-1)     
                                

        max_score = float("-inf")
        max_move_idx = -1

        for idx,move in enumerate(moves):                                 
            new_game = game.forecast_move(move)                          
            next_player = not maximizing_player
            score,_move = self.min_val(new_game, depth-1,alpha, beta, next_player)            
            if (score > alpha):
                alpha = score
            if (score >= beta):                
                return score,_move#check this

            if score > max_score:
                max_score = score
                max_move_idx = idx            

        return max_score,moves[max_move_idx]

    def min_val(self, game, depth, alpha, beta,maximizing_player):
        
        moves = game.get_legal_moves()
		
        if (depth == 0 or len(moves) == 0):                                
            return self.score(game,self),(-1,-1)     
                        
        moves = game.get_legal_moves()        

        min_score = float("inf")
        min_move_idx = -1        

        for idx,move in enumerate(moves):                                 
            new_game = game.forecast_move(move)                          
            next_player = not maximizing_player
            score,_move = self.max_val(new_game, depth-1,alpha, beta, next_player)            
            if (score < beta):
                beta = score
            if (score <= alpha):                
                return score,_move#check this

            if score < min_score:
                min_score = score
                min_move_idx = idx            
		
        
        return min_score,moves[min_move_idx]
        
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
        score,move =  self.max_val( game, depth, alpha, beta,True)
        return move
