import pygame
from State import State
from Graphics import *

class HumanAgent:
    def __init__(self, environment, player= 1) -> None:
        self.player = player
        self.mode = "origin"
        self.origin = None
        self.destination = None
        self.environment = environment

    # modes:
    # origin -  player need to pick a piece.
    # origin -> destination: player picked legal piece (Knight) and need to choose destination.
    # destination -> origin: player picked legal destination
    # origin -> ball: player picked legal ball and need to choose destination.
    # ball -> ball2: player picked legal destination and need to choose again
    # ball2 -> ball: player choose again
    # ball2 -> origin: player choose finish return action: (origin), (destination) 
    
    
    def get_Action(self, events = None, graphics: Graphics = None, state= None):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                time.sleep(0.2)
                pos = pygame.mouse.get_pos()
                row_col = graphics.calc_row_col(pos)
                if self.mode == "origin":
                    if not self.environment.is_legal_origin(self.environment.state, row_col):
                        print("this is an illegal choise")
                        return None
                    self.origin = row_col
                    # print("player ", self.player, ":", self.mode, self.origin, self.destination)
                    self.mode = "destination"
                    return None
                if self.mode == "destination":
                    self.destination = row_col
                    if not self.environment.is_legal_destination(self.environment.state, self.origin, row_col):
                        print("this is an illegal step")
                        return None
                    self.mode = "origin"
                    action = self.origin, self.destination
                    # print("player ", self.player, ":", self.mode, self.origin, self.destination)
                    self.origin, self.destination = None,None
                    return action
               
                return None
            else:
                return None

