import numpy as np
from State import State
from Graphics import *

class Environment:
    def __init__(self, state) -> None:
        self.state = state

    def is_legal_origin (self, state, origin):
        row, col = origin
        if not (row <=6 and row >= 0 and col <=7 and col >= 0):
            return False
        if(state.player == 1):
            return state.board[row, col] == 1 or state.board[row, col] == 3
        else:
            return state.board[row, col] == 2 or state.board[row, col] == 4 
    
    def is_legal_destination(self,state,origin,destination):
        row, col = destination
        o_r, o_c = origin
        
        # out of board
        if row <=6 and row >= 0 and col <=7 and col >= 0:
            # origin is knight and destination is ocupated
            if state.board[origin] == 1 or state.board[origin] == 2:
                if (np.abs(o_c - col) == 1 and np.abs(o_r - row) == 2) or (np.abs(o_c - col) == 2 and np.abs(o_r- row) == 1) and state.board[row,col] == 0:
                    return True
                else:
                    return False
            # origin is ball:
            #   1) destination is the same color and knight
            #   2) destination is in line
            #       a) the same row and different column = true
            #       b) the same column and different row = true;
            #       c) diagonal ABS(origin_row- destination_row) == ABS(origin_col- destination_col)
            #   3) destination is not blocked
            if state.board[origin] == 3 or state.board[origin] == 4:
                if state.board[destination] == state.board[origin] - 2:
                    if o_r == row and o_c != col:
                        if o_c > col:
                            if o_c - col != 1:
                                for i in range(col + 1, o_c):
                                    if state.board[row, i] != state.board[row, o_c] - 2 and state.board[row,i] != 0:
                                        return False
                        if o_c < col:
                            if col - o_c != 1:
                                for i in range(o_c + 1, col):
                                    if state.board[row, i] != state.board[row, o_c] - 2 and state.board[row,i] != 0:
                                        return False
                        return True
                    if o_r != row and o_c == col:
                        if o_r > row:
                            if o_r - row != 1:
                                for i in range(row + 1, o_r):
                                    if state.board[i,col] != state.board[o_r, col] - 2 and state.board[i,col] != 0:
                                        return False
                        if o_r < row:
                            if row - o_r != 1:
                                for i in range(o_r + 1, row):
                                    if state.board[i,col] != state.board[o_r, col] - 2 and state.board[i,col] != 0:
                                        return False
                        return True
                    if np.abs(o_r - row) == np.abs(o_c - col):
                        if o_r > row:
                            if o_c > col:
                                if np.abs(o_r - row) != 1:        
                                    for i in range(np.abs(o_r - row)):
                                        if state.board[row + i + 1, col + i + 1] != state.board[o_r, o_c] - 2 and state.board[row + i + 1, col + i + 1] != 0:
                                            return False
                            if col > o_c:
                                if np.abs(o_r - row) != 1:  
                                    for i in range(np.abs(o_r - row)):
                                        if state.board[row + i + 1, o_c + i + 1] != state.board[o_r, o_c] - 2 and state.board[row + i + 1, o_c + i + 1] != 0:
                                            return False
                            return True
                        if row > o_r:
                            if o_c > col:
                                if np.abs(o_r - row) != 1:
                                    for i in range(np.abs(o_r - row)):
                                        if state.board[o_r + i + 1, col + i + 1] != state.board[o_r, o_c] - 2 and state.board[o_r + i + 1, col + i + 1] != 0:
                                            return False
                            if col > o_c:
                                if np.abs(o_r - row) != 1:
                                    for i in range(np.abs(o_r - row)):
                                        if state.board[o_r + i + 1, o_c + i + 1] != state.board[o_r, o_c] - 2 and state.board[o_r + i + 1, o_c + i + 1] != 0:
                                            return False
                            return True
                    else:
                        return False
                else:
                    return False
        else:
            return False
        
    def end_of_game(self, state: State):
        for i in range(7):
            if state.board[i,7] == 3:
                return 1
        for i in range(7):
            if state.board[i,0] == 4:
                return 2
        return 0
    
    def step(self, state:State, action):
        origin, destination = action
        origin_row, origin_col = origin
        destination_row, destination_col = destination
        state.board[origin_row, origin_col], state.board[destination_row, destination_col] =  state.board[destination_row, destination_col], state.board[origin_row, origin_col]
        state.SwitchPlayer()
    
    def get_next_state (self, state:State, action):
        next_state = state.copy()
        self.step(next_state, action)
        return next_state

    def legal_actions(self, state: State):
        legal_action = []
        squares = np.where(state.board == state.player)
        squares = list(zip(squares[0], squares[1]))
        ball = np.where(state.board == state.player + 2)
        ball = list(zip(ball[0], ball[1]))
        ball = ball[0]
        s_dir = [(1, 2), (2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
        for s in squares:
            for dir in s_dir:
                dest_row, dest_col = s[0] + dir[0], s[1] + dir[1]
                if self.is_legal_destination(state,(s[0],s[1]), (dest_row,dest_col)):
                    legal_action.append(((s[0],s[1]), (dest_row,dest_col)))
        for s in squares:
            if self.is_legal_destination(state, ball, (s[0], s[1])):
                legal_action.append((ball, (s[0], s[1])))
                
                
        return legal_action
    
    def reward (self, state : State, action = None) -> tuple:
        if action:
            next_state = self.get_next_state(state, action)
        else:
            next_state = state
        if (self.end_of_game(next_state)):
            sum =  next_state.board.sum()
            for i in range(7):
                if state.board[i,7] == 3:
                    return 1, True
            for i in range(7):
                if state.board[i,0] == 4:
                    return -1, True
        return 0, False

    def get_init_state(self, Rows_Cols = (ROWS, COLS)):
        rows, cols = Rows_Cols
        board = np.array([[0, 0, 0, 0, 0, 0, 0, 0], 
                         [1, 0, 0 , 0, 0, 0, 0, 2], 
                         [1, 0, 0, 0, 0, 0, 0, 2], 
                         [3, 0, 0, 0, 0, 0, 0, 4], 
                         [1, 0, 0, 0, 0, 0, 0, 2], 
                         [1, 0, 0, 0, 0, 0, 0, 2], 
                         [0, 0, 0, 0, 0, 0, 0, 0]])
        return State (board, player=1)