import pygame
from Graphics import *
from Environment import Environment
from State import State
MAXSCORE = 1000

class AlphaBetaAgent:
    def __init__(self, environment: Environment, player= 1, depth=2) -> None:
        self.player = player
        self.environment = environment
        self.depth = depth
    
    def evaluate (self, state):
        
        BALL_VALUE = 2

        white_pos = np.where(state.board == 1)
        white_ball_pos = np.where(state.board == 3)
        white_free_for_ball = 0
        eof = self.environment.end_of_game(state) 
        if eof == self.player:
            return 2000
        elif eof != 0:
            return -2000
        legal = self.environment.legal_actions(state)
        for i in range (0,len(legal)):
            if state.board[legal[i][0]] == 3:
                white_free_for_ball += legal[i][1][1]
        white_score = np.sum(white_pos[1])+(white_ball_pos[1][0] * BALL_VALUE)+(white_free_for_ball)

        
        
        black_pos = np.where(state.board == 2)
        black_ball_pos = np.where(state.board == 4)
        black_free_for_ball = 0
        legal = self.environment.legal_actions(state)
        for i in range (0,len(legal)):
            if state.board[legal[i][0]] == 4:
                black_free_for_ball += 7 - legal[i][1][1]
        black_score = np.sum(7-black_pos[1])+(7-black_ball_pos[1][0]* BALL_VALUE)+(black_free_for_ball)

        if self.player == 1:
            return white_score- black_score
        else:
            return black_score - white_score    

        
    
    def get_Action(self, events = None, graphics = None, state = None):
        value, bestAction = self.minMax(state)
        return bestAction

    def minMax(self, state:State):
        visited = set()
        depth = 0
        alpha = -MAXSCORE
        beta = MAXSCORE
        return self.max_value(state, visited, depth, alpha, beta)
        
    def max_value (self, state:State, visited:set, depth, alpha, beta):
        
        value = -MAXSCORE

        # stop state
        if depth == self.depth or self.environment.end_of_game(state) != 0:
            value = self.evaluate(state)
            return value, None
        
        # start recursion
        bestAction = None
        legal_actions = self.environment.legal_actions(state)
        for action in legal_actions:
            newState = self.environment.get_next_state(action=action, state=state)
            if newState not in visited:
                visited.add(newState)
                newValue, newAction = self.min_value(newState, visited,  depth + 1, alpha, beta)
                if newValue > value:
                    value = newValue
                    bestAction = action
                    alpha = max(alpha, value)
                if value >= beta:
                    return value, bestAction

        return value, bestAction 

    def min_value (self, state:State, visited:set, depth, alpha, beta):
        
        value = MAXSCORE

        # stop state
        if depth == self.depth or self.environment.end_of_game(state) != 0:
            value = self.evaluate(state)
            return value, None
        
        # start recursion
        bestAction = None
        legal_actions = self.environment.legal_actions(state)
        for action in legal_actions:
            newState = self.environment.get_next_state(action=action, state=state)
            if newState not in visited:
                visited.add(newState)
                newValue, newAction = self.max_value(newState, visited,  depth + 1, alpha, beta)
                if newValue < value:
                    value = newValue
                    bestAction = action
                    beta = min(beta, value)
                if value <= alpha:
                    return value, bestAction

        return value, bestAction 
