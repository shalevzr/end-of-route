import numpy as np
import pygame
from pyparsing import col
import time
from State import State
from constant import *


pygame.init()
white = ((255,255,255))


class Graphics:
    def __init__(self, win):
        # self.board = state.board
        # self.rows, self.cols = self.board.shape
        self.win = win







    def draw_all_pieces(self, state:State):
        board = state.board
        row, col = board.shape
        for row in range(ROWS):
            for col in range(COLS):
                self.draw_piece(state, (row, col))
            
    def draw_piece(self, state:State, row_col):
        row, col = row_col # tuple for pos
        board = state.board # ?
        number = board[row][col] # 0 empty 1 us 2 enemy
        pos = self.calc_base_pos(row_col) # 
        color = self.calc_color(number)
        pygame.draw.rect(self.win, color, (*pos, SQUARE_SIZE-PADDING, SQUARE_SIZE-PADDING))
        if number == 3:
            pygame.draw.circle(self.win,WHITECIRCLE,[pos[0]+50,pos[1]+50],45)
        if number == 4:
            pygame.draw.circle(self.win,BLACKCIRCLE,[pos[0]+50,pos[1]+50],45)    

        
        row, col = row_col
        y = row * SQUARE_SIZE + SQUARE_SIZE//2 + FRAME
        x = col * SQUARE_SIZE + SQUARE_SIZE//2 + FRAME
        return x, y

    def calc_base_pos(self, row_col):
        row, col = row_col
        y = row * SQUARE_SIZE + FRAME
        x = col * SQUARE_SIZE + FRAME
        return x , y

    def calc_num_pos(self, row_col, font, number):
        row, col = row_col
        font_width, font_height = font.size(str(number))
        y = row * SQUARE_SIZE + FRAME + (SQUARE_SIZE - font_height)//2
        x = col * SQUARE_SIZE + FRAME + (SQUARE_SIZE - font_width)//2
        return x, y

    def calc_row_col(self, pos):
        x, y = pos
        col = x // SQUARE_SIZE
        row = y // SQUARE_SIZE
        return row, col

    def calc_color(self, number):
            if number == 0:
                return BROWN
            elif number == 1:
                return WHITE
            elif number == 2:
                return BLACK
            elif number == 3:
                return WHITE
            elif number == 4:
                return BLACK

    def draw(self, state):
        self.win.fill(LIGHTGRAY)
        self.draw_all_pieces(state)

    def draw_square(self, row_col, color):
        pos = self.calc_base_pos(row_col)
        pygame.draw.rect(self.win, color, (*pos, SQUARE_SIZE, SQUARE_SIZE))

    def blink(self, row_col, color):
        row, col = row_col
        player = self.board[row][col]
        for i in range (3):
            self.draw_square((row, col), color)
            pygame.display.update()
            time.sleep(0.2)
            self.draw_piece((row, col))
            pygame.display.update()
            time.sleep(0.2)
    

    

        
    