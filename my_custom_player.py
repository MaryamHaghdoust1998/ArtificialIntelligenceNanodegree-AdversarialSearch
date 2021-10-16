from sample_players import DataPlayer
import random


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """

    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))

        else:
            depth_limit = 6
            for depth in range(depth_limit):
                best_move = self.AlphaBeta(state, depth)
                self.queue.put(best_move)

    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        remaining_own_liberties = sum(len(state.liberties(l)) for l in own_liberties)
        remaining_opp_liberties = sum(len(state.liberties(l)) for l in opp_liberties)
        return len(own_liberties) - 2 * len(opp_liberties) + remaining_own_liberties - 2 * remaining_opp_liberties

    def MinVal(self, state, alpha, beta, depth):
        if state.terminal_test():
            return state.utility(self.player_id)
        if depth <= 0:
            return self.score(state)
        point = float("inf")
        for action_i in state.actions():
            point = min(point, self.MaxVal(state.result(action_i), alpha, beta, depth - 1))
            if point <= alpha:
                return point
            beta = min(beta, point)
        return point

    def MaxVal(self, state, alpha, beta, depth):
        if state.terminal_test():
            return state.utility(self.player_id)
        if depth <= 0:
            return self.score(state)
        point = float("-inf")
        for action_a in state.actions():
            point = max(point, self.MinVal(state.result(action_a), alpha, beta, depth - 1))
            if point >= beta:
                alpha = max(alpha, point)
        return point

    def AlphaBeta(self, state, depth):

        def killer_moves(best_moves, next_move):
            best_move, best_score, alpha = best_moves
            value = self.MinVal(state.result(next_move), alpha, float("-inf"), depth - 1)
            alpha = max(alpha, value)
            if value >= best_score:
                return next_move, value, alpha
            else:
                return best_move, best_score, alpha

        move = (None, float("-inf"), float("-inf"))
        for action in state.actions():
            move = killer_moves(move, action)
        return move[0]


