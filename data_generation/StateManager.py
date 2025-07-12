import numpy as np
from typing import List, Tuple
from data_generation.simulation import SimulationConfig
from data_generation.CourierScheduler import CourierScheduler


class StateManager:
    """Computes and updates the state vector (s_t^7) for the rapid delivery system."""

    def __init__(self, config: SimulationConfig):
        self.config = config

    def compute_state(
        self, t: float, courier_scheduler: CourierScheduler, active_orders: List[Tuple]
    ) -> List[float]:
        """Computes the 7-dimensional state vector (s_t^7) at time t."""
        time_remaining = self.config.H - t
        q_couriers = courier_scheduler.get_active_couriers(t)
        q_orders = len([o for o in active_orders if o[4] is None and o[1] <= t < o[2]])
        Theta1 = courier_scheduler.get_courier_changes(
            t, self.config.state_params["k1"]
        )
        Theta2 = 0
        Theta3 = 0
        Theta4 = 0
        return [time_remaining, q_couriers, q_orders, Theta1, Theta2, Theta3, Theta4]