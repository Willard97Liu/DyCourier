from tqdm import tqdm
import numpy as np
from methods.agent import REINFORCE_Agent
from envs.envs import DynamicQVRPEnv
from data_generation.simulation import SimulationConfig

def train_agents(
    agent_configs = {}
):
    config = SimulationConfig()
    hidden_layers = [1024, 1024, 1024]
    
    def env_fn(seed=None):
        config = SimulationConfig(seed=seed)
        return DynamicQVRPEnv(config)

    agent = REINFORCE_Agent(env_fn, hidden_layers, **agent_configs)
    agent.train(epoch=10000)
    
    
if __name__ == "__main__":
    train_agents()