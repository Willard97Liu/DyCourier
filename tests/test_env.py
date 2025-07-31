import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from methods.agent import DQNAgent  
from envs.envs import DynamicQVRPEnv
from data_generation.config import SimulationConfig
import os, re

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

from pathlib import Path
import os, re

def get_latest_model():
    base_dir = Path(__file__).resolve().parent.parent  # -> QyCourier_test/
    results_dir = base_dir / "results"

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    model_files = [f for f in os.listdir(results_dir) if re.match(r'model_DQN_ep\d+\.pt', f)]
    if not model_files:
        raise FileNotFoundError("No model_DQN_ep*.pt files found in results/")

    model_files.sort(key=lambda f: int(re.findall(r'\d+', f)[0]))
    return str(results_dir / model_files[-1])



def evaluate(model_path, num_episodes=50):
    config = SimulationConfig()
    env = DynamicQVRPEnv(config)
    
    agent = DQNAgent(env, hidden_layers=[1024, 1024, 1024], test=True)
    agent.model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    
    
    agent.model.eval() 
    rewards = agent.test(num_episodes)
    
    for i, r in enumerate(rewards):
        print(f"Episode {i+1}: reward = {r:.2f}")
    
    return np.array(rewards)

if __name__ == "__main__":
    model_path = get_latest_model()
    print("Loading model from:", model_path)
    rewards = evaluate(model_path, num_episodes=5)
    np.save("test_rewards.npy", rewards)
