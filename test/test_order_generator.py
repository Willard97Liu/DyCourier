import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_generation.simulation import SimulationConfig
from data_generation.order_generator import OrderGenerator  # This should be the version with Order objects

def test_order_generator_object_based():
    config = SimulationConfig()
    order_generator = OrderGenerator(config)
    orders = order_generator.get_orders()  # List[Order] objects

    # Extract fields from Order objects into a DataFrame
    df = pd.DataFrame([{
        "release_time": o.order_time,
        "ready_time": o.ready_time,
        "deadline": o.due_time,
        "location": o.location,
        "status": o.status,
        "assigned_courier": o.assigned_courier,
        "delivery_time": o.delivery_time,
    } for o in orders])

    # === Check 1: Orders per location should be in [10, 20] ===
    orders_per_location = df["location"].value_counts().sort_index()

    assert all(
        (10 <= count <= 20) for count in orders_per_location
    ), f"Order count per location out of bounds: {orders_per_location.tolist()}"

    # === Check 2: Total number of orders should be in [160, 320] ===
    total_orders = len(orders)
    assert 160 <= total_orders <= 320, f"Total order count {total_orders} out of expected range."

    # Save orders to CSV for inspection
    df.to_csv("test/csvfile/order_objects.csv", index=False)

    print("Test passed: Order object-based generation is within expected bounds.")

