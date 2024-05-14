import numpy as np
import torch

class State:
    def __init__(self, board=None, player=1):
        if board is None:
            self.board = self.init_board()
        else:
            self.board = board # np.array
        self.rows, self.cols = self.board.shape
        self.player = player

    def init_board (self):
        return np.array([[0, 0, 0, 0, 0, 0, 0, 0], 
                         [1, 0, 0 , 0, 0, 0, 0, 2], 
                         [1, 0, 0, 0, 0, 0, 0, 2], 
                         [3, 0, 0, 0, 0, 0, 0, 4], 
                         [1, 0, 0, 0, 0, 0, 0, 2], 
                         [1, 0, 0, 0, 0, 0, 0, 2], 
                         [0, 0, 0, 0, 0, 0, 0, 0]])


    def get_blank_pos (self):
        pos = np.where(self.board == 0)
        row = pos[0].item()
        col = pos[1].item()
        return row, col

    def SwitchPlayer (self):
        if self.player == 1:
            self.player = 2
        else:
            self.player = 1

    def __eq__(self, other):
        return np.equal(self.board, other.board).all()

    def copy(self):
        newBoard = np.copy(self.board)
        return State (board=newBoard,player=self.player)

    def getcols(self):
        return self.cols

    def getrows(self):
        return self.rows  
    
    def __hash__(self) -> int:
        return hash(repr(self.board) + repr(self.player))
    
    def toTensor (self, device = torch.device('cpu')) -> tuple:
        board_np = self.board.reshape(-1)
        board_tensor = torch.tensor(board_np, dtype=torch.float32, device=device)
        return board_tensor
    
    [staticmethod]
    def tensorToState (state_tensor, player):
       
        board = state_tensor.reshape([7,8]).cpu().numpy()
        return State(board, player=player)
          