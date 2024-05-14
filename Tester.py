from RandomAgent import RandomAgent
from Environment import Environment
from DQN_Agent import DQN_Agent
from State import State
import torch

class Tester:
    def __init__(self, environment, player1, player2) -> None:
        self.environment = environment
        self.player1 = player1
        self.player2 = player2
        

    def test (self, games_num):
        environment = self.environment
        player = self.player1
        player1_win = 0
        player2_win = 0
        games = 0
        while games < games_num:
            print(games, end='\r')
            action = player.get_Action(state=environment.state, train = False)
            environment.step(state = environment.state, action=action )
            player = self.switchPlayers(player)
            reward, end_of_game = environment.reward(environment.state)
            if end_of_game:
                if reward > 0:
                    player1_win += 1
                elif reward < 0:
                    player2_win += 1
                environment.state = environment.get_init_state()
                games += 1
                player = self.player1
        return player1_win, player2_win        

    def switchPlayers(self, player):
        if player == self.player1:
            return self.player2
        else:
            return self.player1

    def __call__(self, games_num):
        return self.test(games_num)

if __name__ == '__main__':
    state = State()
    environment = Environment(state)
    environment.state = environment.get_init_state()
    player1 = DQN_Agent(environment=environment, player=1, train=False)
    params = torch.load('Data/results_2.pth')['params']
    player1.DQN.load_state_dict(params)
    player2 = RandomAgent(environment, player=2)
    test = Tester(environment,player1, player2)
    print(test.test(100))
    