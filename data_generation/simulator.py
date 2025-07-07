import numpy as np
import random
from typing import List
from data_generation.CourierScheduler import CourierScheduler
from data_generation.order_generator import OrderGenerator
from data_generation.StateManager import StateManager
from data_generation.simulation import SimulationUtils, SimulationConfig


# Simulation Logic
class Simulator:
    """Runs the simulation and generates training data for DQN training.

    This class orchestrates the simulation of a rapid delivery system, managing
    orders, couriers, and decision-making to produce state-action-reward-next_state
    tuples. It follows the Markov Decision Process (MDP) framework from the paper,
    using the experimental settings from page 15 (e.g., H=540 minutes, 16 pickup
    locations, courier types {1, 1.5} hours).
    """

    def __init__(self, config: SimulationConfig):
        """Initializes the simulator with configuration and dependencies.

        Args:
            config: SimulationConfig object containing parameters like operating
                period (H), decision interval, courier costs, and state vector settings.
        """
        # Store configuration for simulation parameters
        self.config = config
        # Initialize order generator to create orders with placement times and locations
        self.order_generator = OrderGenerator(config)
        # Initialize courier scheduler to manage base and on-demand couriers
        self.courier_scheduler = CourierScheduler(config)
        # Initialize state manager to compute the 7-dimensional state vector (s_t^7)
        self.state_manager = StateManager(config)
        # Initialize utility functions for order assignment and lost order calculation
        self.utils = SimulationUtils()
        # Define decision epochs (every decision_interval minutes, e.g., every 5 minutes)
        self.decision_epochs = np.arange(
            0, config.H0 + config.decision_interval, config.decision_interval
        )
        # Define action space: combinations of adding 0, 1, or 2 couriers of each type (1-hour, 1.5-hour)
        self.action_space = [
            (a1, a1_5)
            for a1 in range(config.max_couriers_per_type + 1)
            for a1_5 in range(config.max_couriers_per_type + 1)
        ]

    def run_episode(self) -> List[List[float]]:
        """Runs a single simulation episode and returns training data.

        Simulates one episode of the delivery system over the operating period (H0=450 minutes),
        generating state-action-reward-next_state tuples for each decision epoch.
        Actions are chosen randomly to create diverse training data for DQN.

        Returns:
            List of lists, where each inner list contains:
            [state (7 elements), action (2 elements), reward, next_state (7 elements)].
        """
        # Initialize couriers with the base schedule (pre-scheduled couriers)
        active_couriers = self.courier_scheduler.base_schedule[:]
        # Initialize orders with placement, ready, due times, locations, and no assignments
        active_orders = [
            (t, r, d, loc, None) for t, r, d, loc in self.order_generator.get_orders()
        ]
        # Track order assignments (order index, courier start/end times)
        order_assignments = []
        # Store experience tuples for this episode
        data = []

        # Iterate over decision epochs (e.g., every 5 minutes)
        for t in self.decision_epochs:
            # Update active couriers: remove those whose shifts have ended
            active_couriers = [
                (start, end) for start, end in active_couriers if t < end
            ]

            # Update active orders: add new orders placed in the current decision interval
            new_orders = [
                (t_o, r_o, d_o, loc, None)
                for t_o, r_o, d_o, loc in self.order_generator.get_orders()
                if t_o <= t < t_o + self.config.decision_interval
            ]
            active_orders.extend(new_orders)
            
            
            # print(f"分配任务前active_orders{active_orders}")

            # Assign orders to available couriers greedily, prioritizing by due time
            # Updates active_orders with assignments and returns new assignments
            active_orders, new_assignments = self.utils.assign_orders(
                t, active_orders, active_couriers, self.config
            )   
            order_assignments.extend(new_assignments)
            
            # print(f"分配任务后active_orders{active_orders}")
            
            return

            # Remove orders that are either assigned and will be delivered on time
            # or unassigned and past their due time (considered lost)
            active_orders = [
                o
                for i, o in enumerate(active_orders)
                if o[4] is None
                or (o[4] is not None and o[2] > t + self.config.s_p + self.config.s_d)
            ] 

            # Compute current state (s_t^7): [time_remaining, q_couriers, q_orders, Theta1, Theta2, Theta3, Theta4]
            state = self.state_manager.compute_state(
                t, self.courier_scheduler, active_orders
            )

            # Select action randomly for exploration (number of 1-hour and 1.5-hour couriers to add)
            action = random.choice(self.action_space)
            a1, a1_5 = (
                action  # Unpack action: a1 = # of 1-hour couriers, a1_5 = # of 1.5-hour couriers
            )
            new_couriers = [1] * a1 + [1.5] * a1_5  # List of courier types to add

            # Compute reward for the action
            t_next = min(
                t + self.config.decision_interval, self.config.H0
            )  # Next decision epoch or end of order placement
            # Count orders lost between t and t_next if action is taken
            n_lost = self.utils.get_lost_orders(
                t, t_next, active_orders, active_couriers, new_couriers, self.config
            )
            # Reward = penalty for lost orders + cost of adding couriers
            reward = self.config.K_lost * n_lost + sum(
                self.config.K_c[c] for c in new_couriers
            )

            # Apply action: add on-demand couriers (start after delta=5 minutes)
            self.courier_scheduler.add_on_demand_couriers(t, action)
            active_couriers.extend(
                [
                    (t + self.config.delta, t + self.config.delta + c * 60)
                    for c in new_couriers
                ]
            )

            # Re-assign orders with updated courier pool
            active_orders, new_assignments = self.utils.assign_orders(
                t, active_orders, active_couriers, self.config
            )
            order_assignments.extend(new_assignments)

            # Compute next state at t_next
            t_next = min(t + self.config.decision_interval, self.config.H0)
            # Update active couriers for next epoch
            active_couriers = [
                (start, end) for start, end in active_couriers if t_next < end
            ]
            # Update active orders for next epoch
            active_orders = [
                o
                for i, o in enumerate(active_orders)
                if o[4] is None
                or (
                    o[4] is not None
                    and o[2] > t_next + self.config.s_p + self.config.s_d
                )
            ]
            # Compute next state (s_{t+Δ}^7)
            next_state = self.state_manager.compute_state(
                t_next, self.courier_scheduler, active_orders
            )

            # Store experience tuple: [state, action, reward, next_state]
            data.append(state + list(action) + [reward] + next_state)

        return data
