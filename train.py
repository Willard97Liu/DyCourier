from tqdm import tqdm
import numpy as np
from methods.agent import DQNAgent
from envs.envs import DynamicQVRPEnv
from data_generation.simulation import SimulationConfig

def train_agents(
    agent_configs = {}
):
    config = SimulationConfig()
    env = DynamicQVRPEnv(config)
    hidden_layers = [1024, 1024, 1024]
    agent = DQNAgent(env, hidden_layers, **agent_configs)
    agent.train(episodes=10000)
    
    
if __name__ == "__main__":
    train_agents()