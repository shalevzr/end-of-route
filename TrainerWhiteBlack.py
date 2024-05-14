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

File_Num = 10

# path_Save=f'Data/params_{File_Num}.pth'
# path_best = f'Data/best_params_{File_Num}.pth'
buffer_path = f'Data/buffer_{File_Num}.pth'
results_path=f'Data/results_{File_Num}.pth'
# random_results_path = f'Data/random_results_{File_Num}.pth'
# path_best_random = f'Data/best_random_params_{File_Num}.pth'


def main ():
    
    player1 = DQN_Agent(player=1, environment=environment,parametes_path=None)
    player1_hat = DQN_Agent(player=1, environment=environment, train=False)
    Q1 = player1.DQN
    Q1_hat = Q1.copy()
    Q1_hat.train = False
    player1_hat.DQN = Q1_hat

    player2 = DQN_Agent(player=2, environment=environment,parametes_path=None)
    player2_hat = DQN_Agent(player=2, environment=environment, train=False)
    Q2 = player2.DQN
    Q2_hat = Q2.copy()
    Q2_hat.train = False
    player2_hat.DQN = Q2_hat
    
    buffer1 = ReplayBuffer(path=None) 
    buffer2 = ReplayBuffer(path=None) 
    
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
    optim1 = torch.optim.Adam(Q1.parameters(), lr=learning_rate)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optim1,10000*100, gamma=0.50)
    optim2 = torch.optim.Adam(Q2.parameters(), lr=learning_rate)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optim2,10000*100, gamma=0.50)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,[30*50000, 30*100000, 30*250000, 30*500000], gamma=0.5)
    
    for epoch in range(start_epoch, epochs):
        # print(f'epoch = {epoch}', end='\r')
        state_1 = environment.get_init_state()
        step = 0
        action_2 = None
        while not environment.end_of_game(state_1) or step > max_step:
            # Sample Environement
            print(step, end='\r')
            step += 1
            action_1 = player1.get_Action(state_1, epoch=epoch)
            after_state_1 = environment.get_next_state(state=state_1, action=action_1)
            reward_1, end_of_game_1 = environment.reward(after_state_1)
            if action_2:
                buffer2.push(state_2, action_2, reward_1, after_state_1, end_of_game_1)
            if end_of_game_1 or step > max_step:
                res += reward_1
                buffer1.push(state_1, action_1, reward_1, after_state_1, True)
                buffer2.push(state_2, action_2, reward_1, after_state_1, True)
                break
            state_2 = after_state_1
            action_2 = player2.get_Action(state=state_2)
            after_state_2 = environment.get_next_state(state=state_2, action=action_2)
            reward_2, end_of_game_2 = environment.reward(state=after_state_2)
            if end_of_game_2 or step > max_step:
                res += reward_2
            buffer1.push(state_1, action_1, reward_2, after_state_2, end_of_game_2)
            buffer2.push(state_2, action_2, reward_1, after_state_1, end_of_game_1)
            state_1 = after_state_2

            if len(buffer1) < MIN_Buffer:
                continue
            
            # Train White NN
            states, actions, rewards, next_states, dones = buffer1.sample(batch_size)
            Q_values = Q1(states, actions)
            next_actions = player1_hat.get_Actions(next_states, dones) 
            with torch.no_grad():
                Q_hat_Values = Q1_hat(next_states, next_actions) 
            loss = Q1.loss(Q_values, rewards, Q_hat_Values, dones)
            loss.backward()
            optim1.step()
            optim1.zero_grad()
            scheduler1.step()
            
             # Train Black NN                   
            states, actions, rewards, next_states, dones = buffer2.sample(batch_size)
            Q_values = Q2(states, actions)
            next_actions = player1_hat.get_Actions(next_states, dones) 
            with torch.no_grad():
                Q_hat_Values = Q2_hat(next_states, next_actions) 
            loss = Q1.loss(Q_values, rewards, Q_hat_Values, dones)
            loss.backward()
            optim2.step()
            optim2.zero_grad()
            scheduler2.step()


            if loss_count <= 1000:
                avgLoss = (avgLoss * loss_count + loss.item()) / (loss_count + 1)
                loss_count += 1
            else:
                avgLoss += (loss.item()-avgLoss)* 0.00001 
            
        if epoch % C == 0:
            Q1_hat.load_state_dict(Q1.state_dict())
            Q2_hat.load_state_dict(Q2.state_dict())
        
        print (f'epoch={epoch} loss={loss:.5f} step={step} avgloss={avgLoss:.5f}', end=" ")
        print (f'learning rate={scheduler1.get_last_lr()[0]} path={results_path} res= {res} best_res = {best_res}')

        
        if (epoch+1) % 100 == 0:
            print(f'\nres= {res}')
            avgLosses.append(avgLoss)
            results.append(res)
            if best_res < res:      
                best_res = res
            res = 0

        if (epoch+1) % 1000 == 0:
            test = tester(100)
            test_score = test[0]-test[1]
            if best_random < test_score and tester(1) == (1,0):
                best_random = test_score
                best_random = player1.DQN.state_dict().copy()
            print(test)
            random_results.append(test_score)

        if (epoch+1) % 50 == 0:
            torch.save({'epoch': epoch, 'results': results, 'avglosses':avgLosses, 'params': player1.DQN.state_dict(), 
                        'best_random': best_random, 'random_results': random_results }, results_path)
            torch.save(buffer1, buffer_path)
            torch.save({'epoch': epoch, 'results': results, 'avglosses':avgLosses, 'params': player2.DQN.state_dict(), 
                        'best_random': best_random, 'random_results': random_results }, results_path)
            torch.save(buffer2, buffer_path)
            # player1.save_param(path_Save)
            # torch.save(random_results, random_results_path)
        
        
    torch.save({'epoch': epoch, 'results': results, 'avglosses':avgLosses, 'params': player1.DQN.state_dict()}, results_path)
    torch.save(buffer1, buffer_path)
    torch.save({'epoch': epoch, 'results': results, 'avglosses':avgLosses, 'params': player2.DQN.state_dict()}, results_path)
    torch.save(buffer2, buffer_path)

if __name__ == '__main__':
    main()