import numpy as np
from tqdm import tqdm
import multiprocess as mp
from copy import deepcopy

import torch
from pathlib import Path


import pickle

from methods.ML.RL import train_DQN, NN



class Agent:
    def __init__(self, env):
        self.env = env
    
    def act(self, **kwargs):
        return int(self.env.action_space.sample())
    
    def test(self, n):
        
        ACTIONS = [
            (0, 0), (1, 0), (0, 1),
            (2, 0), (1, 1), (0, 2)
        ]
        episode_rewards = np.zeros(n)
        
        
        BASE_DIR = Path(__file__).resolve().parent.parent
        
        # 骑手的测试数据
        courier_PATH = BASE_DIR / "data_generation" / "test_data" / "base_courier.pkl"
        with open(courier_PATH, "rb") as f:
            base_courier_data = pickle.load(f)
        # 订单的测试数据
        ORDER_TEST_PATH = BASE_DIR / "data_generation" / "test_data" / "base_orders.pkl"
        with open(ORDER_TEST_PATH, "rb") as f:
            base_order_data = pickle.load(f)
            
        
        for i in tqdm(range(n), desc="Evaluating DQN Agent"):
            self.env.set_mode("test")
        
            self.env.courier_scheduler.base_schedule = base_courier_data[i].copy()
            
            self.env.active_orders = [
                (float(t), float(r), float(d), int(loc), None, None)
                for t, r, d, loc in base_order_data[i]
            ]
            
            state = self.env.reset()
            episode_reward = 0
            episode_lost = 0
            episode_order = 0
            

            for t in self.env.decision_epochs:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.model(state_tensor)
                action_id = q_values.argmax(dim=1).item()
                action = ACTIONS[action_id]

                reward, next_state, lost_n = self.env.step(t, action)
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
        hidden_layers,
        test = False,
        **kwargs
        ):
        super().__init__(env, **kwargs)
        
        if test:
            self.model = NN(
                7,
                hidden_layers,
                6
            )
            
        self.env =env
        self.hidden_layers = hidden_layers
        
        
    def train(self, episodes, **kwargs):
        
        train_DQN(
            env = self.env,
            hidden_layers = self.hidden_layers,
            num_episodes = episodes
        )
        