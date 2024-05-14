import pygame
from State import State
from Graphics import *
import random

class  RandomAgent:
    def __init__(self, environment, player= 1) -> None:
        self.player = player
        self.environment = environment

     
    def get_Action(self, events = None, graphics: Graphics = None, state= None, train = False):
        legal_actions = self.environment.legal_actions(state)
        return random.choice(legal_actions)