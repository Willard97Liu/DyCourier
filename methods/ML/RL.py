import numpy as np
from copy import deepcopy

import math
import random
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt


from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback


class NN(nn.Module):#(BaseFeaturesExtractor):

    def __init__(self, 
                #  observation_space: spaces.Box, 
                 n_observation, 
                 hidden_layers = [512, 512, 256],
                 n_actions: int = 1):
        super().__init__()
        hidden = deepcopy(hidden_layers)
        hidden.insert(0, n_observation)
        layers = []
        for l in range(len(hidden)-1):
            layers += [
                nn.Linear(hidden[l], hidden[l+1]),
                nn.ReLU()
            ]
        layers += [
            nn.Linear(hidden[-1], n_actions),
            # nn.Sigmoid()
            # nn.Softmax()
        ]

        self.linear = nn.Sequential(
            *layers
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.linear(state)
    
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, 
                 n_observation, 
                 hidden_layers = [512, 512, 256],
                 n_actions: int = 2):
        
        super().__init__()
        
        hidden_layers.insert(0, n_observation)
        layers = []
        for l in range(len(hidden_layers)-1):
            layers += [
                nn.Linear(hidden_layers[l], hidden_layers[l+1]),
                nn.ReLU()
            ]
            
        layers += [
            nn.Linear(hidden_layers[-1], n_actions),
        ]

        self.linear = nn.Sequential(
            *layers
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(observations)
    

def train_DQN(
    env,
    LR = 1e-4,
    BATCH_SIZE = 1,
    hidden_layers = [1024, 1024, 1024],
    EPISODES = 20,
    update_target_every = 100,
    TAU = 0.01,
):
    num_episodes = EPISODES
    len_state = 7
    ACTIONS = [
    (0, 0),
    (1, 0),
    (0, 1),
    (2, 0),
    (1, 1),
    (0, 2)]
    
    n_actions = len(ACTIONS)
    
    policy_net = NN(len_state, deepcopy(hidden_layers), n_actions)
    target_net = NN(len_state, deepcopy(hidden_layers), n_actions)
    
    global steps_done
    steps_done = 0
    
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)
    
    def select_action(state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        else:
            state = state.float()
            
        state = state.unsqueeze(0)  # 增加 batch 维度，变成 [1, state_dim]
        action_id = policy_net(state).max(1).indices.view(1, 1)
        return action_id
    
    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        optimizer.zero_grad()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()
    
    for i_episode in range(num_episodes):
        
        env.reset()
        active_orders = [] 
        for t in env.decision_epochs:
            active_couriers = [
                (start, end) for start, end in env.active_couriers if t < end
            ]
            new_orders = [
                (t_o, r_o, d_o, loc, None)
                for t_o, r_o, d_o, loc in env.order_generator.get_orders()
                if t_o <= t < t_o + env.config.decision_interval
            ]
            env.active_orders.extend(new_orders)
            
            
            # 分配订单, 看顾客和订单的时间，然后将订单分配给顾客
            env.active_orders, _ = env.utils.assign_orders(
                t, env.active_orders, env.active_couriers, env.config
            )
            
            # 减去没用的订单
            env.active_orders = [
                o
                for i, o in enumerate(env.active_orders)
                if o[4] is None
                or (o[4] is not None and o[2] > t + env.config.s_p + env.config.s_d)
            ]
            
            state = env.state_manager.compute_state(
                t, env.courier_scheduler, env.active_orders
            )
            
            
            # # 分析状态
            action_id = select_action(state)
            action = ACTIONS[action_id] 
            reward, next_state = env.step(t, action)
            
            # next_state = torch.tensor(next_state, dtype=torch.float32)
            # action = torch.tensor(action_id.item())
            # memory.push(state, action, reward, next_state)
            # # print(f"Step {t}, memory size = {len(memory)}")

            # # Move to the next state
            # state = next_state

            # # Perform one step of the optimization (on the policy network)
            # optimize_model()
            
            # target_net_state_dict = target_net.state_dict()
            # policy_net_state_dict = policy_net.state_dict()
                
                
            # if i_episode%update_target_every == 0:
            #     # Periodic hard update of the target network's weights
                
            #     for key in policy_net_state_dict:
            #         target_net_state_dict[key] = policy_net_state_dict[key]
            #     target_net.load_state_dict(target_net_state_dict)
            # else:
                
            #     for key in policy_net_state_dict:
            #         target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            #     target_net.load_state_dict(target_net_state_dict)


    print('Complete')       
            
    return 
    
                
                

