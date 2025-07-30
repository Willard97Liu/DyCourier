import sys
import os
import csv
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from data_generation.simulator import Simulator
from data_generation.simulation import SimulationConfig
from data_generation.CourierScheduler import CourierScheduler


def test_run__courier_scheduler_and_export_csv():
    # Initialize config and scheduler
    config = SimulationConfig()
    courier_scheduler = CourierScheduler(config)

    # Initial base couriers
    base_couriers = courier_scheduler.base_schedule
    assert len(base_couriers) > 0, "Base couriers should be initialized"

    # Add a known number of on-demand couriers
    t = 100  # example decision epoch time
    action = (2, 1)  # Add 2 one-hour couriers and 1 1.5-hour courier
    courier_scheduler.add_on_demand_couriers(t, action)

    # Check if they were added
    on_demand = courier_scheduler.on_demand_schedule
    assert len(on_demand) == 3, "3 on-demand couriers should have been added"

    # Get full courier schedule with IDs
    full_schedule = courier_scheduler.full_schedule
    assert len(full_schedule) == len(base_couriers) + 3

    # Export to CSV
    csv_path = "couriers.csv"
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["courier_id", "start_time", "end_time"])
        for courier_id, start, end in full_schedule:
            writer.writerow([courier_id, start, end])

    # Re-load and validate the CSV
    df = pd.read_csv(csv_path)
    assert df.shape[0] == len(full_schedule), "CSV should have same number of rows as couriers"
    assert "courier_id" in df.columns
    assert df["courier_id"].is_unique, "Courier IDs should be unique"
