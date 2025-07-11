import numpy as np
import random
from typing import List
from data_generation.CourierScheduler import CourierScheduler
from data_generation.order_generator import OrderGenerator
from data_generation.StateManager import StateManager
from data_generation.simulation import SimulationUtils, SimulationConfig


class Simulator:
    """Runs the simulation and generates training data for DQN training.

    Orchestrates the simulation of a rapid delivery system, managing orders, couriers, and decisions
    to produce state-action-reward-next_state tuples for Deep Q-Network (DQN) training. Implements
    the Markov Decision Process (MDP) from the paper 'Dynamic Courier Capacity Acquisition in Rapid
    Delivery Systems: A Deep Q-Learning Approach' (page 15), simulating over H0=450 minutes with
    decision epochs every 5 minutes. Orders are retained until past their due time (t < d_o),
    simplifying state management and ensuring unassigned orders contribute to q_orders during peak
    periods (180–300, 360–450 minutes). The delivery duration (s_p + t_travel + s_d = 28 minutes)
    limits assignments, addressing the q_orders=0 issue by retaining unassigned orders.
    """

    def __init__(self, config: SimulationConfig):
        """Initializes the simulator with configuration and dependencies.

        Sets up components for order generation, courier scheduling, state computation, and utility
        functions. Defines decision epochs and action space per the paper’s experimental settings
        (page 15), supporting realistic delivery durations with travel time (t_travel=20).

        Args:
            config: SimulationConfig object with parameters like H0=450, decision_interval=5,
                    s_p=4, s_d=4, t_travel=20, K_lost=-1, max_couriers_per_type=2.
        """
        # Store the configuration object for access to simulation parameters
        # Includes operating period (H0=450), service times (s_p=4, s_d=4), travel time (t_travel=20),
        # lost order penalty (K_lost=-1), and max couriers per type (2)
        self.config = config
        # Initialize order generator to create orders with placement times (t_o in [0, 450]),
        # ready times (r_o = t_o + 10), due times (d_o = t_o + 40), and locations (0 to 15)
        self.order_generator = OrderGenerator(config)
        # Initialize courier scheduler for base and on-demand couriers
        # Base: D1 ~ U[20,30] (1-hour shifts), D1.5 ~ U[10,20] (1.5-hour shifts)
        self.courier_scheduler = CourierScheduler(config)
        # Initialize state manager to compute the 7-dimensional state vector (s_t^7)
        # Components: time_remaining (H - t), q_couriers, q_orders, Theta1, Theta2, Theta3, Theta4
        self.state_manager = StateManager(config)
        # Initialize utility functions for greedy order assignment and lost order calculation
        # Handles assignment logic and reward computation (lost orders and courier costs)
        self.utils = SimulationUtils()
        # Define decision epochs from 0 to H0 (450 minutes) in steps of 5 minutes
        # Creates 91 epochs (0, 5, 10, ..., 450), as specified on page 15
        self.decision_epochs = np.arange(
            0, config.H0 + config.decision_interval, config.decision_interval
        )
        # Define action space as combinations of adding 0, 1, or 2 couriers of each type (1-hour, 1.5-hour)
        # Results in 9 actions: (0,0), (0,1), (0,2), (1,0), ..., (2,2)
        self.action_space = [
            (a1, a1_5)
            for a1 in range(config.max_couriers_per_type + 1)
            for a1_5 in range(config.max_couriers_per_type + 1)
        ]

    def run_episode(self) -> List[List[float]]:
        """Runs a single simulation episode and returns training data.

        Simulates one episode over H0=450 minutes, generating state-action-reward-next_state tuples
        for each decision epoch (every 5 minutes). Manages orders and couriers, assigns orders greedily,
        and computes rewards based on lost orders and courier costs. Orders are kept until past their
        due time (t < d_o), simplifying retention logic and ensuring unassigned orders contribute to
        q_orders. The delivery duration (s_p + t_travel + s_d = 4 + 20 + 4 = 28 minutes) in
        assign_orders limits assignments due to the tight due time window (d_o = t_o + 40), helping
        maintain non-zero q_orders during peak demand periods (180–300, 360–450 minutes), addressing
        the q_orders=0 issue observed in previous data.

        Returns:
            List of lists, where each inner list contains:
            - state: 7 elements (time_remaining, q_couriers, q_orders, Theta1, Theta2, Theta3, Theta4)
            - action: 2 elements (number of 1-hour and 1.5-hour couriers added)
            - reward: Float (K_lost * n_lost + sum of courier costs)
            - next_state: 7 elements (same as state, for t_next)
        """
        # Initialize active couriers with the base schedule from CourierScheduler
        active_couriers = self.courier_scheduler.base_schedule[:]
        # Initialize active orders with all orders generated for the episode
        active_orders = [
            (t, r, d, loc, None, None)
            for t, r, d, loc in self.order_generator.get_orders()
        ]
        # Initialize list for DQN training tuples: [state (7), action (2), reward (1), next_state (7)]
        data = []

        # Iterate over decision epochs (t = 0, 5, 10, ..., 450)
        for t in self.decision_epochs:
            # Remove couriers whose shifts are not active at t
            active_couriers = [
                (start, end) for start, end in active_couriers if start <= t < end
            ]
            # Assign orders greedily using SimulationUtils.assign_orders
            active_orders = self.utils.assign_orders(
                t, active_orders, active_couriers, self.config
            )
            # Filter unassigned orders (assigned field is None)
            unassigned_orders = [o for o in active_orders if o[4] is None]
            # Filter assigned orders that have not yet been delivered (t < delivered_time)
            assigned_but_have_not_delivered = [
                o for o in active_orders if o[4] is not None and t < o[5]
            ]
            # Active orders are the union of unassigned and assigned-but-not-delivered orders
            active_orders = unassigned_orders + assigned_but_have_not_delivered

            # Compute state vector (s_t^7) at time t
            state = self.state_manager.compute_state(
                t, self.courier_scheduler, active_orders
            )
            # Select random action for exploration: (a1, a1_5) where a1, a1_5 in {0, 1, 2}
            action = random.choice(self.action_space)
            # Unpack action into number of 1-hour and 1.5-hour couriers
            a1, a1_5 = action
            # Create list of new courier types (e.g., [1, 1, 1.5] for a1=2, a1_5=1)
            new_couriers = [1] * a1 + [1.5] * a1_5
            # Calculate next epoch time (t + 5, capped at H0=450)
            t_next = min(t + self.config.decision_interval, self.config.H0)
            # Count lost orders between t and t_next
            n_lost = self.utils.get_lost_orders(
                t, t_next, active_orders, active_couriers, new_couriers, self.config
            )
            # Compute reward
            reward = self.config.K_lost * n_lost + sum(
                self.config.K_c[c] for c in new_couriers
            )

            # Add on-demand couriers starting at t + delta (5 minutes)
            self.courier_scheduler.add_on_demand_couriers(t, action)
            # Add new couriers to active list
            active_couriers.extend(
                [
                    (t + self.config.delta, t + self.config.delta + c * 60)
                    for c in new_couriers
                ]
            )
            # Re-assign orders with updated courier pool
            active_orders = self.utils.assign_orders(
                t, active_orders, active_couriers, self.config
            )
            # Update active orders for next epoch: keep unassigned or not-yet-delivered orders not past due
            unassigned_orders = [
                o for o in active_orders if o[4] is None and t_next < o[2]
            ]
            assigned_but_have_not_delivered = [
                o
                for o in active_orders
                if o[4] is not None and t < o[5] and t_next < o[2]
            ]
            active_orders = unassigned_orders + assigned_but_have_not_delivered

            # Compute next state at t_next
            next_state = self.state_manager.compute_state(
                t_next, self.courier_scheduler, active_orders
            )
            # Store experience tuple
            data.append(state + list(action) + [reward] + next_state)

            # Update couriers for next epoch
            active_couriers = [
                (start, end) for start, end in active_couriers if start <= t_next < end
            ]

        # Return experience tuples for DQN training
        return data


# The reward problem, sum on the overall time interval or just one time interval?
# The threshold
# Deep learning in transportation science
