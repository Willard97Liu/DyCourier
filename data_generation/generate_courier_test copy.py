import os
import pickle
import numpy as np
import os,sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from simulation import SimulationConfig
from typing import List, Tuple

def generate_test_schedules(config: SimulationConfig, seed_list: List[int], save_path: str):
    test_schedules = []

    for seed in seed_list:
        rng = np.random.default_rng(seed)
        D1 = rng.integers(config.D1_range[0], config.D1_range[1] + 1)
        D1_5 = rng.integers(config.D1_5_range[0], config.D1_5_range[1] + 1)
        schedule = []

        for t in [60, 120, 270, 330]:
            count = (D1 // 6) if t in [60, 120] else (D1 // 3)
            schedule.extend([(max(0, t + rng.integers(-20, 21)), 1) for _ in range(count)])

        for t in [0, 120, 240, 360]:
            count = D1_5 // 6
            schedule.extend([(max(0, t + rng.integers(-20, 21)), 1.5) for _ in range(count)])

        test_schedules.append(schedule)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(test_schedules, f)

    print(f"✅ Saved {len(test_schedules)} schedules to {save_path}")

if __name__ == "__main__":
    config = SimulationConfig()  # 初始化你的配置
    seed_list = list(range(100))
    save_path = "data_generation/test_data/base_courier.pkl"
    generate_test_schedules(config, seed_list, save_path)
