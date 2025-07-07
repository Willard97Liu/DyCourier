import numpy as np
from typing import List, Tuple
from data_generation.simulation import SimulationConfig
from data_generation.CourierScheduler import CourierScheduler


# State Management
class StateManager:
    """Computes and updates the state vector (s_t^7) for the rapid delivery system.

    This class is responsible for generating the 7-dimensional value-based state vector
    (s_t^7) as defined in the paper, which captures the system's status at each decision
    epoch. The state includes time remaining, active couriers, active orders, and vectors
    encoding future courier changes, past order placements, and orders at risk of being late.
    It uses the experimental settings from page 15 (e.g., H=540 minutes, k1=120, k2=30, k3=40).
    """

    def __init__(self, config: SimulationConfig):
        """Initializes the StateManager with simulation configuration.

        Args:
            config: SimulationConfig object containing parameters like operating period (H),
                service times (s_p, s_d), and state vector parameters (k1, k2, k3).
        """
        # Store configuration for access to parameters like H, s_p, s_d, and state_params
        self.config = config

    def compute_state(
        self, t: float, courier_scheduler: CourierScheduler, active_orders: List[Tuple]
    ) -> List[float]:
        """Computes the 7-dimensional state vector (s_t^7) at time t.

        The state vector is defined as s_t = (H - t, q_t^couriers, q_t^orders, Theta_t^1, Theta_t^2, Theta_t^3, Theta_t^4),
        where each component provides information for the DQN to decide on adding on-demand couriers.
        The state is computed based on the current time, courier schedules, and active orders.

        Args:
            t: Current time in minutes within the operating period.
            courier_scheduler: CourierScheduler object to query active couriers and schedule changes.
            active_orders: List of tuples (t_o, r_o, d_o, loc, assigned) representing orders,
                where t_o is placement time, r_o is ready time, d_o is due time, loc is location,
                and assigned is the courier index (or None if unassigned).

        Returns:
            List of 7 floats: [time_remaining, q_couriers, q_orders, Theta1, Theta2, Theta3, Theta4].
        """
        # Time remaining until the end of the operating period (H - t)
        # Captures temporal demand patterns, as later times may have fewer orders
        time_remaining = self.config.H - t

        # Number of active couriers at time t (couriers whose shifts include t)
        # Reflects current delivery capacity
        q_couriers = courier_scheduler.get_active_couriers(t)

        # Number of active orders at time t (unassigned orders that are ready but not past due)
        # Indicates current workload
        q_orders = len([o for o in active_orders if o[4] is None and o[1] <= t < o[2]])

        # Theta1: Net change in courier count in the next k1 minutes (e.g., k1=120)
        # Counts couriers scheduled to start minus those scheduled to end in (t, t+k1]
        # Helps predict future delivery capacity
        Theta1 = courier_scheduler.get_courier_changes(
            t, self.config.state_params["k1"]
        )

        # Theta2: Number of orders placed in the past k2 minutes (e.g., k2=30)
        # Reflects recent demand trends to inform capacity needs
        Theta2 = sum(
            1
            for t_o, _, _, _, _ in active_orders
            if t - self.config.state_params["k2"] < t_o <= t
        )

        # Theta3: Number of orders at risk of being late in the next k3 minutes (e.g., k3=40)
        # Counts unassigned orders where pickup in (t, t+k3] would result in late delivery
        # Indicates near-term capacity shortages
        Theta3 = sum(
            1
            for _, r_o, d_o, _, _ in active_orders
            if t < max(r_o, t + self.config.s_p) <= t + self.config.state_params["k3"]
            and max(r_o, t + self.config.s_p) + self.config.s_p + self.config.s_d > d_o
        )

        # Theta4: Average lateness of orders at risk of being late in (t, t+k3]
        # Measures the average time by which at-risk orders would miss their due time
        # Complements Theta3 by quantifying severity of potential delays
        Theta4 = (
            np.mean(
                [
                    d_o
                    - (
                        max(r_o, t + self.config.s_p)
                        + self.config.s_p
                        + self.config.s_d
                    )
                    for _, r_o, d_o, _, _ in active_orders
                    if t
                    < max(r_o, t + self.config.s_p)
                    <= t + self.config.state_params["k3"]
                    and max(r_o, t + self.config.s_p)
                    + self.config.s_p
                    + self.config.s_d
                    > d_o
                ]
            )
            if Theta3 > 0
            else 0
        )

        # Return the state vector as a list
        return [time_remaining, q_couriers, q_orders, Theta1, Theta2, Theta3, Theta4]
