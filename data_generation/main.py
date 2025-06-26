import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime
from typing import List
from data_generation.simulator import Simulator
from data_generation.simulation import SimulationConfig


# Data Output
def save_to_csv(data: List[List[float]], config: SimulationConfig) -> str:
    """Saves simulation data to a CSV file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"training_data_{timestamp}.csv"
    columns = [
        "time_remaining",
        "q_couriers",
        "q_orders",
        "Theta1",
        "Theta2",
        "Theta3",
        "Theta4",
        "action_1hr",
        "action_1.5hr",
        "reward",
        "next_time_remaining",
        "next_q_couriers",
        "next_q_orders",
        "next_Theta1",
        "next_Theta2",
        "next_Theta3",
        "next_Theta4",
    ]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(csv_filename, index=False)
    return csv_filename


# Main Execution
def main():
    config = SimulationConfig()
    simulator = Simulator(config)
    all_data = []
    for _ in range(10):  # Run 10 episodes
        episode_data = simulator.run_episode()
        all_data.extend(episode_data)
    csv_filename = save_to_csv(all_data, config)
    print(f"Data saved to {csv_filename}")


if __name__ == "__main__":
    main()
