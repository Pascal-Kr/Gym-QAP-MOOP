from gym_flp.envs.flp_env_2D_Umgebung_Mehrziel import qapEnv
from Deep_Q_learning_komplett_neu_Versuch_Torch_final import DQNAgent
import os
import torch as T
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime


current_time = datetime.now().strftime('%b%d_%H-%M')
writer = SummaryWriter(log_dir=os.path.join('logs_pytorch_Test', current_time + '_Rücklauf100_1k'))


if __name__ == '__main__':
    env=qapEnv(mode ='rgb_array', instance='Neos-n7')
    ename='Lärm100_100k'
    Aggregate_Stats_Every=50
    Probeläufe = 200
    MHCmin, MHCmax, Rücklaufmin, Rücklaufmax, Lärmmin, Lärmmax, Lärmpunktzahlmin, Lärmpunktzahlmax = env.Probelauf(Probeläufe)
    average_reward=0
    load_checkpoint = True
    write_Logfile = False
    Show_Actions = False
    Build_Average = False
    Aggregate_Stats_Every = 10
    episodes=10
    if write_Logfile == True:
        episodes = 1000
    
    # For stats
    ep_rewards = []
    ep_Minimums = []
    ep_last_Costs = []
    ep_last_Rückläufe = []
    ep_last_Lärmpunktzahl = []


    agent = DQNAgent(episodes, gamma=0.8, epsilon=1, lr=0.0001,
                     input_dims=(env.observation_space_values),
                     n_actions=env.action_space.n, Replay_Memory_Size=5000, eps_min=0,
                     batch_size=64, Update_Target_Every=1000,
                     chkpt_dir='models/', algo='DQNAgent',
                     env_name=ename)
    

    if load_checkpoint:
        agent.load_models()
        agent.epsilon=0


    for episode in range(1, episodes + 1):
        done = False
        current_state = env.reset()
        if Show_Actions == True:
            env.render()
        if write_Logfile == False:
            print('Anfangs-Lärmpunktzahl: ' + str(env.initial_Lärmpunktzahl))
            print('Anfangs-MHC: ' + str(int(env.initial_MHC)))
            print('Anfangs-Rückläufe: ' + str(env.initial_Rückläufe))
        episode_reward = 0
        while not done:
            action = agent.choose_action(current_state)
            new_state, reward, done, info = env.step(action)
            episode_reward += reward
            if Show_Actions == True:
                env.render()
            if write_Logfile == False:
                print('Lärmpunktzahl: ' + str(info['Lärm']))
                print('MHC: ' + str(int(info['MHC'])))
                print('Rückläufe: ' + str(info['Rückläufe']))
                print('MHC Reward: ' + str(env.MHCreward))
                print('Rücklaufreward: ' + str(env.Rücklaufreward))
                print('Lärm Reward: ' + str(env.Lärmreward))
                print('Lärm Reward Intervalle: ' + str(env.Lärmrewardintervalle))
                print('Reward für Aktion: ' + str(reward))
                print('Episoden Reward: ' + str(episode_reward))
                print('')

            current_state = new_state
        
        ep_rewards.append(episode_reward)
        #ep_Minimums.append(env.Actual_MHCmin)
        ep_last_Costs.append(info['MHC'])
        ep_last_Rückläufe.append(info['Rückläufe'])
        ep_last_Lärmpunktzahl.append(info['Lärm'])
        if write_Logfile == True:
            writer.add_scalar('Episode Reward', episode_reward, episode)
            writer.add_scalar('Last MHC', info['MHC'], episode)
            writer.add_scalar('Last Rückläufe', info['Rückläufe'],episode)
            writer.add_scalar('Last Lärmpunktzahl', info['Lärm'],episode)
        if not episode % Aggregate_Stats_Every and Build_Average == True:
            average_reward = sum(ep_rewards[-Aggregate_Stats_Every:])/len(ep_rewards[-Aggregate_Stats_Every:])
            writer.add_scalar('Average Reward', average_reward, episode)
            
Minimum_MHC = min(ep_last_Costs)          
Maximum_MHC = max(ep_last_Costs)
Minimum_Rückläufe = min(ep_last_Rückläufe)
Maximum_Rückläufe = max(ep_last_Rückläufe)
Minimum_Lärmpunktzahl = min(ep_last_Lärmpunktzahl)
Maximum_Lärmpunktzahl = max(ep_last_Lärmpunktzahl)
            
