import numpy as np
from copy import deepcopy

import math
import random
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

import sys
import os
from pathlib import Path
path = Path(os.path.dirname(__file__))
sys.path.insert(1, str(path.parent.absolute()))



results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)  # 如果不存在就创建



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
    BATCH_SIZE = 64,
    hidden_layers = [1024, 1024, 1024],
    num_episodes = 20,
    update_target_every = 100,
    TAU = 0.01,
    GAMMA = 0.99,
    eval_interval = 2,
    save = True,
    model_path = 'model_DQN',
):
    
    device = torch.device(
        # "cuda" if torch.cuda.is_available() else
        # "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    
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
            
        state = state.unsqueeze(0) 
        action_id = policy_net(state).squeeze(1).max(1).indices.view(1, 1)
        return action_id
    

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        
        with torch.no_grad():
            next_state_values = target_net(next_states).max(1).values
        
        next_states = torch.cat([s for s in batch.next_state])
        
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        
        # Compute L2 loss
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        optimizer.zero_grad()
        
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()
        
        
    for i_episode in range(num_episodes):
        
        state = env.reset()
        
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        for t in env.decision_epochs:
            
            action_id = select_action(state)
            
            action = ACTIONS[action_id]
            
            
            reward, next_state = env.step(t, action)
            
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            
            reward = np.array(reward, np.float32)
            
            reward = torch.tensor(reward[None])
            
            memory.push(state, action_id, next_state, reward)
            # # Move to the next state
            state = next_state
            
            # # Perform one step of the optimization (on the policy network)
            optimize_model()
            
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            
            if i_episode%update_target_every == 0:
                # Periodic hard update of the target network's weights
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]
                target_net.load_state_dict(target_net_state_dict)
            else:
                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                target_net.load_state_dict(target_net_state_dict)
            
            
            if t == 450 and (i_episode + 1) % eval_interval == 0:
                # test_r = eval()
                # mean_r = test_r.mean()
                # print(f"{i_episode + 1} : {mean_r:.3f}")
                
                # if mean_r > best_r:
                #     best_r = mean_r
                #     print(f"best! mean rewards: {best_r:.3f}")
                if save:
                    # print(f"dkfjdkfjfdk")
                    torch.save(policy_net.state_dict(), results_dir/f'model_{model_path}')
                    # print(f"Model saved to: {os.path.abspath(model_path)}")

                    
                
    return 
    
                
                

