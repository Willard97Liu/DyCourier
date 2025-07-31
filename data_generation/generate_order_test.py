import os
import pickle
from typing import List
from simulation import SimulationConfig
from order_generator import OrderGenerator  # 确保模块名正确
import numpy as np
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def generate_order_episodes(config: SimulationConfig, seeds: List[int], save_path: str):
    all_episodes = []

    for seed in seeds:
        np.random.seed(seed)
        generator = OrderGenerator(config)
        orders = generator.get_orders()

        # 强制类型转换为纯 Python 类型（避免 numpy.float64）
        episode_orders = [
            (float(t_o), float(r_o), float(d_o), int(loc))
            for (t_o, r_o, d_o, loc) in orders
        ]

        all_episodes.append(episode_orders)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(all_episodes, f)

    print(f"✅ Saved {len(all_episodes)} episodes to {save_path}")


if __name__ == "__main__":
    config = SimulationConfig()
    seeds = list(range(100))
    save_path = "data_generation/test_data/base_orders.pkl"
    generate_order_episodes(config, seeds, save_path)
