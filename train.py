from tqdm import tqdm
import numpy as np
from methods.agent import RLAgent
from envs.envs import DynamicQVRPEnv
from data_generation.simulation import SimulationConfig

def train_agents(
    agent_configs = {}
):
    config = SimulationConfig()
    env = DynamicQVRPEnv(config)
    agent = RLAgent(env, algo='DQN',**agent_configs)
    agent.train(episodes=15)
    
    
if __name__ == "__main__":
    train_agents()