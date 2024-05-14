import torch
import random
import math
import numpy as np
from DQN import DQN
from constant import *
from State import State


class DQN_Agent:
    def __init__(self, player = 1, parametes_path = None, train = True, environment= None):
        self.DQN = DQN()
        if parametes_path:
            self.DQN.load_params(parametes_path)
        self.player = player
        self.train = train
        self.setTrainMode()
        self.environment = environment

    def setTrainMode (self):
          if self.train:
              self.DQN.train()
          else:
              self.DQN.eval()

    def get_Action (self, state:State, epoch = 0, events= None, train = True, graphics = None, black_state = None) -> tuple:
        actions = self.environment.legal_actions(state)
        if self.train and train:
            epsilon = self.epsilon_greedy(epoch)
            rnd = random.random()
            if rnd < epsilon:
                return random.choice(actions)
        
        state_tensor = state.toTensor()
        expand_state_tensor = state_tensor.unsqueeze(0).repeat((len(actions),1))
        actions_tensor = torch.tensor(np.array(actions))
        actions_tensor = actions_tensor.reshape(-1, 4)
        with torch.no_grad():
            Q_values = self.DQN(expand_state_tensor, actions_tensor)
        max_index = torch.argmax(Q_values)
        return actions[max_index]

    def get_Actions (self, states_tensor: State, dones) -> torch.tensor:
        actions = []
        for i, state_tensor in enumerate(states_tensor):
            if dones[i].item():
                actions.append(((0,0),(0,0)))
            else:
                actions.append(self.get_Action(state=State.tensorToState(state_tensor=state_tensor,player=self.player), train=False))
        action_tensor = torch.tensor(actions).reshape(-1,4)
        return action_tensor

    def epsilon_greedy(self,epoch, start = epsilon_start, final=epsilon_final, decay=epsiln_decay):
        if epoch > decay:
            return epsilon_final
        return start - (start - final) * epoch/decay
        
    
    def loadModel (self, file):
        self.model = torch.load(file)['params']
    
    def save_param (self, path):
        self.DQN.save_params(path)

    def load_params (self, path):
        self.DQN.load_params(path)

    def __call__(self, events= None, state=None):
        return self.get_Action(state)