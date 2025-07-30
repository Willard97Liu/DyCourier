import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_generation.simulator import Simulator
from data_generation.simulation import SimulationConfig, SimulationUtils
from data_generation.order_generator import OrderGenerator
from data_generation.CourierScheduler import CourierScheduler
from data_generation.StateManager import StateManager

def test_run__episode():
    config = SimulationConfig()
    courier_scheduler = CourierScheduler(config)
    order_generator = OrderGenerator(config)
    utils = SimulationUtils()
    state_manager = StateManager(config)
    order_before_assign = []
    order_after_assign = []
    current_orders_rows = []
    all_state_rows = []
    all_courier_rows = []
    active_orders = [
        (t, r, d, loc, None, None) for t, r, d, loc in order_generator.get_orders()
    ]
    # Log raw active orders
    for o in active_orders:
        order_before_assign.append(
            {
                "release_time": o[0],
                "ready_time": o[1],
                "deadline": o[2],
                "location": o[3],
                "assigned_courier": o[4],
                "pickup_time": o[5],
            }
        )


    decision_epochs = np.arange(
        0, config.H0 + config.decision_interval, config.decision_interval
    )

    for t in decision_epochs:
        # Filter active couriers
        active_couriers = [
            (i, s, e) for i, s, e in courier_scheduler.base_schedule if s <= t < e
        ]
        # print('time: ',t, "Active couriers", active_couriers)
        # Log courier data
        for i, s, e in active_couriers:
            all_courier_rows.append({"time": t, "index": i, "start": s, "end": e})
        courier_number = courier_scheduler.get_active_couriers(t)
        print("time:", t, "number of courier:", courier_number, "index", [row["index"] for row in all_courier_rows if row["time"] == t])

        current_orders = [
            o
            for o in active_orders
            if o[0] <= t and (o[4] is None or t < o[5]) and t < o[2]
        ]
        for o in current_orders:
            current_orders_rows.append(
                {
                    "time": t,
                    "release_time": o[0],
                    "ready_time": o[1],
                    "deadline": o[2],
                    "location": o[3],
                    "assigned_courier": o[4],
                    "pickup_time": o[5],
                }
            )

        # Assign orders
        active_orders = utils.assign_orders(t, active_orders, active_couriers, config)
        # Log raw active orders
        for o in active_orders:
            order_after_assign.append(
                {
                    "time": t,
                    "release_time": o[0],
                    "ready_time": o[1],
                    "deadline": o[2],
                    "location": o[3],
                    "assigned_courier": o[4],
                    "pickup_time": o[5],
                }
            )
        # Compute and log state
        state = state_manager.compute_state(t, courier_scheduler, active_orders)
        # Check the number of courier
        assert state[1] == courier_number 
        # Check the Theta_1
        
        all_state_rows.append({"timestamp": t, "state": state})

    # Save to CSV
    pd.DataFrame(order_before_assign).to_csv("order_before_assign.csv", index=False)
    pd.DataFrame(current_orders_rows).to_csv("current_orders.csv", index=False)
    pd.DataFrame(order_after_assign).to_csv("orders_after_assign.csv", index=False)
    pd.DataFrame(all_state_rows).to_csv("output_states.csv", index=False)
    pd.DataFrame(all_courier_rows).to_csv("output_couriers.csv", index=False)
