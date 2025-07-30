import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_generation.simulation import SimulationConfig
from data_generation.order_generator import OrderGenerator

def test_order_generator():
    config = SimulationConfig()
    order_generator = OrderGenerator(config)
    orders = order_generator.get_orders()

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(
        orders,
        columns=["release_time", "ready_time", "deadline", "location"]
    )

    # === Check 1: Orders per location should be in [10, 20] ===
    orders_per_location = df["location"].value_counts().sort_index()

    assert all(
        (10 <= count <= 20) for count in orders_per_location
    ), f"Order count per location out of bounds: {orders_per_location.tolist()}"

    # === Check 2: Total number of orders should be in [160, 320] ===
    # This range is based on 16 locations × [10, 20] orders per location
    total_orders = len(orders)
    assert 160 <= total_orders <= 320, f"Total order count {total_orders} out of expected range."

    # Save raw orders to CSV
    df["assigned_courier"] = None
    df["pickup_time"] = None
    df.to_csv("order.csv", index=False)

    print("Test passed: Order distribution is within expected bounds.")
