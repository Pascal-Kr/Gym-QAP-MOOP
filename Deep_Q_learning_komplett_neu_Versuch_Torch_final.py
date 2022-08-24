from collections import deque
import random
from gym_flp.envs.flp_env_2D_Umgebung_Mehrziel import qapEnv
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

use_cuda = T.cuda.is_available()
current_time = datetime.now().strftime('%b%d_%H-%M')

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, new_state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states_, terminal


class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        
        self.conv1 = nn.Conv2d(input_dims[0], 32, 2, stride=1)
        self.conv2 = nn.Conv2d(32, 64, 2, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        #conv3 = F.relu(self.conv3(conv2))
        # conv3 shape is BS x n_filters x H x W
        conv_state = conv2.view(conv2.size()[0], -1)
        # conv_state shape is BS x (n_filters * H * W)
        flat1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(flat1)

        return actions

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
    
    


class DQNAgent(object):
    def __init__(self, episodes, chkpt_dir, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min,
                 replace=100, algo=None, env_name=None):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.loss_tracker=0
        self.running_loss=0
        
        self.episodes=episodes
        self.EPSILON_START = 1
        self.EPSILON_END_1 = 0.1
        self.EPISODES_END_1 = 0.6 * self.episodes #Nach 60% der Episoden 10% Exploration
        self.EPISODES_END_2 = self.episodes - self.EPISODES_END_1
        self.EPSILON_DECAY_LIN_1 = (self.EPSILON_START-self.EPSILON_END_1)/self.EPISODES_END_1 #lineare Senkung von Epsilon bis EPISODES_END_1
        self.EPSILON_DECAY_LIN_2 = (self.EPSILON_END_1-self.eps_min)/(self.EPISODES_END_2-self.episodes*0.05) #lineare Senkung von Epsilon bis EPISODES_END_2

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_eval',
                                    chkpt_dir=self.chkpt_dir)

        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_next',
                                    chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation],dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = \
                                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
    
    def decrement_epsilon(self):
        if self.epsilon >= self.EPSILON_END_1:     
            self.epsilon -= self.EPSILON_DECAY_LIN_1
        elif self.epsilon >= self.eps_min:
            self.epsilon -= self.EPSILON_DECAY_LIN_2
        if self.epsilon <= self.eps_min:
            self.epsilon = self.eps_min
        return self.epsilon

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        self.loss_tracker = loss.item()
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        #self.decrement_epsilon()
    


if __name__ == '__main__':
    env=qapEnv(mode ='rgb_array', instance='Neos-n7')
    ename='Rücklauf100_20k'
    writer = SummaryWriter(log_dir=os.path.join('logs_pytorch', current_time + ename))
    Aggregate_Stats_Every=50
    Probeläufe = 20000
    MHCmin, MHCmax, Rücklaufmin, Rücklaufmax, Lärmmin, Lärmmax, Lärmpunktzahlmin, Lärmpunktzahlmax = env.Probelauf(Probeläufe)
    average_reward=0
    best_score = -np.inf
    load_checkpoint = False
    episodes=20000
    
    # For stats
    ep_rewards = []
    ep_Minimums = []
    ep_Costs = []
    ep_Rückläufe = []
    ep_Lärmpunktzahl = []


    agent = DQNAgent(episodes, gamma=0.8, epsilon=1, lr=0.0001,
                     input_dims=(env.observation_space_values),
                     n_actions=env.action_space.n, mem_size=5000, eps_min=0.001,
                     batch_size=64, replace=1000,
                     chkpt_dir='models/', algo='DQNAgent',
                     env_name=ename)

    if load_checkpoint:
        agent.load_models()

    #fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' \
     #       + str(episodes) + 'games'
    #figure_file = 'plots/' + fname + '.png'
    # if you want to record video of your agent playing, do a mkdir tmp && mkdir tmp/dqn-video
    # and uncomment the following 2 lines.
    #env = wrappers.Monitor(env, "tmp/dqn-video",
    #                    video_callable=lambda episode_id: True, force=True)
    n_steps = 0
    episode_rewards, eps_history, steps_array = [], [], []

    for episode in tqdm(range(1, episodes + 1), ascii=True, unit='episodes'):
        done = False
        current_state = env.reset()

        episode_reward = 0
        while not done:
            action = agent.choose_action(current_state)
            new_state, reward, done, info = env.step(action)
            episode_reward += reward

            if not load_checkpoint:
                agent.store_transition(current_state, action,
                                     reward, new_state, done)
                agent.learn()
            current_state = new_state
            n_steps += 1
        loss = agent.loss_tracker
        writer.add_scalar('loss',loss,episode)
        epsilon = agent.decrement_epsilon()
        episode_rewards.append(episode_reward)
        steps_array.append(n_steps)
        
        ep_rewards.append(episode_reward)
        #ep_Minimums.append(env.Actual_MHCmin)
        ep_Costs.append(info['MHC'])
        ep_Rückläufe.append(info['Rückläufe'])
        ep_Lärmpunktzahl.append(info['Lärm'])
        if not episode % Aggregate_Stats_Every:
            average_reward = sum(ep_rewards[-Aggregate_Stats_Every:])/len(ep_rewards[-Aggregate_Stats_Every:])
            min_reward = min(ep_rewards[-Aggregate_Stats_Every:])
            #Minimum_costs = min(ep_Minimums[-Aggregate_Stats_Every:])
            max_reward = max(ep_rewards[-Aggregate_Stats_Every:])
            average_MHC = sum(ep_Costs[-Aggregate_Stats_Every:])/len(ep_Costs[-Aggregate_Stats_Every:])
            average_Rückläufe = sum(ep_Rückläufe[-Aggregate_Stats_Every:])/len(ep_rewards[-Aggregate_Stats_Every:])
            average_Lärmpunktzahl = sum(ep_Lärmpunktzahl[-Aggregate_Stats_Every:])/len(ep_rewards[-Aggregate_Stats_Every:])
            #average_end_costs = sum(ep_Costs[-AGGREGATE_STATS_EVERY:])/len(ep_Costs[-AGGREGATE_STATS_EVERY:])
            writer.add_scalar('Average Reward', average_reward, episode)
            writer.add_scalar('Average MHC', average_MHC, episode)
            writer.add_scalar('Average Rückläufe', average_Rückläufe,episode)
            writer.add_scalar('Average Lärmpunktzahl', average_Lärmpunktzahl,episode)
            writer.add_scalar('Epsilon', epsilon,episode)
       
            # Save model
            if episode % 2500 == 0: 
                agent.save_models()




