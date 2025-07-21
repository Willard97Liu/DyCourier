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
        q_orders = len([o for o in active_orders if o[4] is None and o[0] <= t < o[2]])
        Theta1 = courier_scheduler.get_courier_changes(
            t, self.config.state_params["k1"]
        )
        Theta2 = sum(
            1
            for t_o, _, _, _, _, _ in active_orders
            if t - self.config.state_params["k2"] < t_o <= t
        )
        Theta3 = sum(
            1
            for _, r_o, d_o, _, _, _ in active_orders
            if t < max(r_o, t + self.config.s_p) <= t + self.config.state_params["k3"]
            and max(r_o, t + self.config.s_p) + self.config.t_travel + self.config.s_d
            > d_o
        )

        Theta4 = (
            np.mean(
                [
                    d_o
                    - (
                        max(r_o, t + self.config.s_p)
                        + self.config.s_p
                        + self.config.t_travel
                        + self.config.s_d
                    )
                    for _, r_o, d_o, _, _, _ in active_orders
                    if t
                    < max(r_o, t + self.config.s_p)
                    <= t + self.config.state_params["k3"]
                    and max(r_o, t + self.config.s_p)
                    + self.config.t_travel
                    + self.config.s_d
                    > d_o
                ]
            )
            if Theta3 > 0
            else 0
        )
        return [time_remaining, q_couriers, q_orders, Theta1, Theta2, Theta3, Theta4]
