from Environment import Environment
from State import State
from DQN_Agent import DQN_Agent
from ReplayBuffer import ReplayBuffer
from RandomAgent import RandomAgent
from AlphaBetaAgent import AlphaBetaAgent
from MinMaxAgent import MinMaxAgent
import torch
from Tester import Tester

epochs = 2000000
start_epoch = 0
C = 350
learning_rate = 0.01
batch_size = 56
state = State()
environment = Environment(state)
MIN_Buffer = 4000

File_Num = 4

# path_Save=f'Data/params_{File_Num}.pth'
# path_best = f'Data/best_params_{File_Num}.pth'
buffer_path = f'Data/buffer_{File_Num}.pth'
results_path=f'Data/results_{File_Num}.pth'
# random_results_path = f'Data/random_results_{File_Num}.pth'
# path_best_random = f'Data/best_random_params_{File_Num}.pth'


def main ():
    
    player1 = DQN_Agent(player=2, environment=environment,parametes_path=None)
    player_hat = DQN_Agent(player=2, environment=environment, train=False)
    Q = player1.DQN
    Q_hat = Q.copy()
    Q_hat.train = False
    player_hat.DQN = Q_hat
    player2 = RandomAgent(environment=environment, player=1)   
    buffer = ReplayBuffer(path=None) # None
    
    results_file = [] #torch.load(results_path)
    results = [] #results_file['results'] # []
    avgLosses = [] #results_file['avglosses']     #[]
    avgLoss = 0 #avgLosses[-1] #0
    loss =0
    res = 0
    best_res = -200
    loss_count = 0
    tester = Tester(player1=player1, player2=RandomAgent(environment=environment, player=-1), environment=environment)
    random_results = [] #torch.load(random_results_path)   # []
    best_random = 0 #max(random_results)
    max_step = 200
    
    # init optimizer
    optim = torch.optim.Adam(Q.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optim,10000*100, gamma=0.50)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,[30*50000, 30*100000, 30*250000, 30*500000], gamma=0.5)
    
    for epoch in range(start_epoch, epochs):
        # print(f'epoch = {epoch}', end='\r')
        state_1 = environment.get_init_state()
        action_1 = player2.get_Action(state=state_1)
        state_1 = environment.get_next_state(state=state_1, action=action_1)     
        step = 1
        while not environment.end_of_game(state_1) or step > max_step:
            # Sample Environement
            print(step, end='\r')
            step += 1
            action_1 = player1.get_Action(state_1, epoch=epoch)
            after_state_1 = environment.get_next_state(state=state_1, action=action_1)
            reward_1, end_of_game_1 = environment.reward(after_state_1)
            reward_1 *= -1
            if end_of_game_1 or step > max_step:
                res += reward_1
                buffer.push(state_1, action_1, reward_1, after_state_1, True)
                break
            state_2 = after_state_1
            action_2 = player2.get_Action(state=state_2)
            after_state_2 = environment.get_next_state(state=state_2, action=action_2)
            reward_2, end_of_game_2 = environment.reward(state=after_state_2)
            reward_2 *= -1
            if end_of_game_2 or step > max_step:
                res += reward_2
            buffer.push(state_1, action_1, reward_2, after_state_2, end_of_game_2)
            state_1 = after_state_2

            if len(buffer) < MIN_Buffer:
                continue
            
            # Train NN
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            Q_values = Q(states, actions)
            next_actions = player_hat.get_Actions(next_states, dones) 
            with torch.no_grad():
                Q_hat_Values = Q_hat(next_states, next_actions) 

            loss = Q.loss(Q_values, rewards, Q_hat_Values, dones)
            loss.backward()
            optim.step()
            optim.zero_grad()
            
            scheduler.step()
            if loss_count <= 1000:
                avgLoss = (avgLoss * loss_count + loss.item()) / (loss_count + 1)
                loss_count += 1
            else:
                avgLoss += (loss.item()-avgLoss)* 0.00001 
            
        if epoch % C == 0:
                Q_hat.load_state_dict(Q.state_dict())

        print (f'epoch={epoch} loss={loss:.5f} step={step} avgloss={avgLoss:.5f}', end=" ")
        print (f'learning rate={scheduler.get_last_lr()[0]} path={results_path} res= {res} best_res = {best_res}')

        
        if (epoch+1) % 100 == 0:
            print(f'\nres= {res}')
            avgLosses.append(avgLoss)
            results.append(res)
            if best_res < res:      
                best_res = res
                # if best_res > 75 and tester(1) == (1,0):
                #     player1.save_param(path_best)
            res = 0

        # if (epoch+1) % 1000 == 0:
        #     test = tester(100)
        #     test_score = test[0]-test[1]
        #     if best_random < test_score and tester(1) == (1,0):
        #         best_random = test_score
        #         player1.save_param(path_best_random)
        #     print(test)
        #     random_results.append(test_score)

        if (epoch+1) % 50 == 0:
            torch.save({'epoch': epoch, 'results': results, 'avglosses':avgLosses, 'params': player1.DQN.state_dict()}, results_path)
            torch.save(buffer, buffer_path)
            # player1.save_param(path_Save)
            # torch.save(random_results, random_results_path)
        
        
    torch.save({'epoch': epoch, 'results': results, 'avglosses':avgLosses, 'params': player1.DQN.state_dict()}, results_path)
    torch.save(buffer, buffer_path)

if __name__ == '__main__':
    main()