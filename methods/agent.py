import numpy as np
from tqdm import tqdm
import multiprocess as mp
from copy import deepcopy

import torch
from torch import nn


from methods.ML.RL import train_DQN, NN

from gymnasium import Env





class Agent:
    def __init__(self, env : Env,
                 n_workers = 2):
        self.env = env
        self.n_workers = n_workers
    
    def act(self, x, *args, **kwargs):
        return int(self.env.action_space.sample())
    
    
    def _simple_run(self, n, initial_instance = 0):
        
        episode_rewards = np.zeros(n)
        actions = [[] for _ in range(n)]
        infos = [[] for _ in range(n)]
        
        for i in tqdm(range(n)):
            o, info = self.env.reset(initial_instance + i)
            infos[i].append(info)
            while True:
                a = self.act(o)
                o, r, d, trun, info = self.env.step(a)
                episode_rewards[i] += r
                actions[i].append(a)
                infos[i].append(info)
                if d or trun:
                    break
        
        return episode_rewards, actions, infos
    
    def run(self, n, initial_instance = 0):
        if self.parallelize:
            return self._parallel_run(n, initial_instance)
        else:
            return self._simple_run(n, initial_instance)
                
    def train(self, episodes):
        pass
    
    
    
class DQNAgent(Agent):
    """The reinforcement learning agent
    This agent is trained with trial and error.
    """
    def __init__(
        self,
        env : Env,
        hidden_layers = [1024, 1024, 1024],
        algo = 'DQN', 
        **kwargs
        ):
        super().__init__(env, **kwargs)
        
        self.model = NN(
            2,
            hidden_layers,
            6
        )
        self.hidden_layers = hidden_layers
        self.algo = algo
            
    def train(self, episodes = 1000, **kwargs):
        
        train_DQN(
            self.env,
            hidden_layers = self.hidden_layers,
            num_episodes = episodes
        )
        

RLAgent = DQNAgent