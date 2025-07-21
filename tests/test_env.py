import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from methods.agent import DQNAgent  
from envs.envs import DynamicQVRPEnv
from data_generation.simulation import SimulationConfig

def evaluate(model_path, num_episodes=50):
    config = SimulationConfig()
    env = DynamicQVRPEnv(config)
    
    agent = DQNAgent(env, hidden_layers=[1024, 1024, 1024])
    agent.model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    agent.model.eval() 
    
    rewards = agent.run(num_episodes)
    
    for i, r in enumerate(rewards):
        print(f"Episode {i+1}: reward = {r:.2f}")
    
    return np.array(rewards)

if __name__ == "__main__":
    model_path = "results/model_DQN"  
    rewards = evaluate(model_path, num_episodes=5)
    np.save("test_rewards.npy", rewards)
