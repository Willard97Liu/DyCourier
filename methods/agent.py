import numpy as np
from tqdm import tqdm
import multiprocess as mp
from copy import deepcopy

import torch
from torch import nn


from methods.ML.RL import train_DQN, NN

from gymnasium import Env





class Agent:
    def __init__(self, env):
        self.env = env
    
    def act(self, x, *args, **kwargs):
        return int(self.env.action_space.sample())
    
    def run(self, n):
        
        ACTIONS = [
            (0, 0), (1, 0), (0, 1),
            (2, 0), (1, 1), (0, 2)
        ]
        episode_rewards = np.zeros(n)
        for i in tqdm(range(n), desc="Evaluating DQN Agent"):
            state = self.env.reset()
            episode_reward = 0

            for t in self.env.decision_epochs:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.model(state_tensor)
                action_id = q_values.argmax(dim=1).item()
                action = ACTIONS[action_id]

                reward, next_state = self.env.step(t, action)
                state = next_state
                episode_reward += reward


            episode_rewards[i] = episode_reward

        return episode_rewards

    
    
class DQNAgent(Agent):
    """The reinforcement learning agent
    This agent is trained with trial and error.
    """
    def __init__(
        self,
        env,
        hidden_layers = [1024, 1024, 1024],
        algo = 'DQN', 
        **kwargs
        ):
        super().__init__(env, **kwargs)
        
        self.model = NN(
            7,
            hidden_layers,
            6
        )
        self.hidden_layers = hidden_layers
        
        
    def train(self, episodes = 1000, **kwargs):
        
        train_DQN(
            self.env,
            hidden_layers = self.hidden_layers,
            num_episodes = episodes
        )
        

RLAgent = DQNAgent