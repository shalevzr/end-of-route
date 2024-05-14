import numpy as np
import pygame
from Graphics import Graphics
from constant import *
from State import State
from AlphaBetaAgent import AlphaBetaAgent
from MinMaxAgent import MinMaxAgent
from HumanAgent import HumanAgent
from RandomAgent import RandomAgent
from Environment import Environment
from DQN_Agent import DQN_Agent
import torch
import time


state = State()
environment = Environment(state)
checkpoint = torch.load('Data/results_4.pth')
#player1 = DQN_Agent(environment = environment, player=1, train=False, parametes_path=None)

#player1.DQN.load_state_dict(checkpoint['params'])
player1 = RandomAgent(environment=environment)
player2 = DQN_Agent(environment = environment, player=2, train=False, parametes_path=None)

player2.DQN.load_state_dict(checkpoint['params'])
#player2 = MinMaxAgent(environment, player=2)

def main ():
    player = player1
    FPS = 60
    win = pygame.display.set_mode((820, 720))
    pygame.display.set_caption('gilad markman the king give me 100 please')
    

    graphics = Graphics(win)

    run = True
    clock = pygame.time.Clock()

    pygame.display.update()

    while(run):

        clock.tick(FPS)
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
               run = False

            time.sleep(0.02)
        action = player.get_Action(events=events, graphics=graphics, state=state)
        if action:
            print(player.player, action)
            environment.step(environment.state, action)
            # if(player.mode == "origin"):
            player = switchPlayer(player)
        graphics.draw(state)
        pygame.display.update() 

        if environment.end_of_game(environment.state) != 0:
            print(f"player number {environment.end_of_game(environment.state)} is the winner")
            time.sleep(2)
            run = False
        
    pygame.quit()

def switchPlayer (player):
    # environment.state.SwitchPlayer() # to delete when changing state
    if player == player1:
        return player2
    else:
        return player1
    
if __name__ == '__main__':
    main()
    