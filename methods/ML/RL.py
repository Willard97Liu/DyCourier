import numpy as np
import math
import random
from collections import namedtuple, deque
from itertools import count
import torch.nn as nn
import torch
import torch.optim as optim
from methods.ML.model import NN
from copy import deepcopy

import sys
import os
from pathlib import Path
from glob import glob
path = Path(os.path.dirname(__file__))
sys.path.insert(1, str(path.parent.absolute()))

result_path = path.parent.parent  # 表示当前脚本的上上一层目录
results_dir = result_path / "results"
results_dir.mkdir(parents=True, exist_ok=True)


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
    
    
def load_latest_model(policy_net, model_path_prefix):
    model_files = sorted(
        glob(str(results_dir / f"{model_path_prefix}_ep*.pt")),
        key=lambda x: int(x.split("_ep")[-1].split(".pt")[0]),
        reverse=True,
    )
    if model_files:
        latest_file = model_files[0]
        print(f"Loading model: {latest_file}")
        policy_net.load_state_dict(torch.load(latest_file))
        return int(latest_file.split("_ep")[-1].split(".pt")[0])
    return 0
    

def train_DQN(
    env,
    hidden_layers,
    num_episodes,
    
    LR = 1e-4,
    BATCH_SIZE = 64,
    update_target_every = 5,
    TAU = 0.01,
    GAMMA = 0.99,
    eval_interval = 50,
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
    
    EPS_START = 1
    EPS_END = 0.05
    EPS_DECAY = 200
    steps_done = 0

    
    start_episode = 0
    if save:
        start_episode = load_latest_model(policy_net, model_path)
        target_net.load_state_dict(policy_net.state_dict())
    
        
    def select_action(state, deterministic=False):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        # print(steps_done)

        if sample > eps_threshold or deterministic:
            with torch.no_grad():
                return policy_net(state).squeeze(1).max(1).indices.view(1, 1)
        else:
            # exploration
            random_action = random.randrange(len(ACTIONS))  # 随机选择一个动作索引
            return torch.tensor([[random_action]], dtype=torch.long)
        
    

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
        state_action_values = policy_net(state_batch).gather(1, action_batch)
        
        

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        
        
        with torch.no_grad():
            next_state_values = target_net(next_states).max(1).values
        
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        
        # Compute L2 loss
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        optimizer.zero_grad()
        
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()
        # print(f"[Loss] {loss.item():.6f}")
        
    
        
    for i_episode in range(start_episode, start_episode + num_episodes):
        
        env.reset()
        
        t = 5
        
        visible_orders = [o for o in env.active_orders if o.order_time <= t]
        
            
        state = env.state_manager.compute_state(
                    t, env.courier_scheduler, visible_orders
                )
        
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        episode_reward = 0 
        
        for t in env.decision_epochs:
            
            action_id = select_action(state)
            
            action = ACTIONS[action_id]
            
            reward, n_lost = env.step(t, action, visible_orders)
            
            reward = np.array(reward, np.float32)
            
            reward = torch.tensor(reward[None])
            
            episode_reward += reward.item()  
            
            t_next = t + env.config.decision_interval
            
        
            next_state = env.state_manager.compute_state(
                    t, env.courier_scheduler, visible_orders
                )
            
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            
            memory.push(state, action_id, next_state, reward)
            
            # print(f"[Push] state: {state}, action_id: {action_id}, next_state: {next_state}, reward: {reward}")
            # # Move to the next state
            state = next_state
            # # Perform one step of the optimization (on the policy network)
            optimize_model()
            
            
            if t == 450 and (i_episode + 1) % eval_interval == 0:
                if save:
                    torch.save(policy_net.state_dict(), results_dir / f'{model_path}_ep{i_episode+1}.pt')
            
            # 下一轮订单
            visible_orders = [o for o in env.active_orders if o.order_time <= t_next]
           
              
        print(f"[Episode {i_episode}] Total reward: {episode_reward}, lost_order: {n_lost}")
        
    
                
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
            
        
    
    
            
    return 
    
                
                

