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
        # Contains (start, end) tuples for 1-hour (60 minutes) and 1.5-hour (90 minutes) shifts
        # Copied to avoid modifying the original schedule
        active_couriers = self.courier_scheduler.base_schedule[:]
        # Initialize active orders with all orders generated for the episode
        # Each order is a tuple (t_o, r_o, d_o, loc, assigned), where:
        # - t_o: Placement time (in [0, 450])
        # - r_o: Ready time (t_o + 10)
        # - d_o: Due time (t_o + 40)
        # - loc: Pickup location index (0 to 15)
        # - assigned: None (unassigned initially)
        active_orders = [
            (t, r, d, loc, None) for t, r, d, loc in self.order_generator.get_orders()
        ]
        # Initialize list to track assignments as (order_idx, courier_start, courier_end) tuples
        # Used for record-keeping and debugging
        order_assignments = []
        # Initialize list for DQN training tuples: [state (7), action (2), reward (1), next_state (7)]
        data = []

        # Iterate over decision epochs (t = 0, 5, 10, ..., 450)
        for t in self.decision_epochs:
            # Remove couriers whose shifts are not active at t
            # Keeps couriers where start <= t < end, ensuring they have started (start <= t)
            # and haven’t ended (t < end), per the paper’s shift availability (page 15)
            active_couriers = [
                (start, end) for start, end in active_couriers if start <= t < end
            ]
            # Assign orders greedily using SimulationUtils.assign_orders
            # Uses t (current decision epoch) to compute pickup_time = max(r_o, t + s_p), where r_o is the order’s ready time
            # Checks deliverability: pickup_time + s_p + t_travel + s_d <= d_o (total 28 minutes from pickup)
            # Updates active_orders (sets assigned field to courier index) and returns new assignments
            active_orders, new_assignments = self.utils.assign_orders(
                t, active_orders, active_couriers, self.config
            )
            # Add new assignments to tracking list for record-keeping
            order_assignments.extend(new_assignments)

            # Filter active_orders to keep only those not past their due time (t < d_o)
            # Applies to both unassigned (o[4] is None) and assigned (o[4] is not None) orders
            # Simplification rationale:
            # - The original condition (o[4] is None and t < o[2] or o[4] is not None and t < o[2])
            #   was redundant, as o[4] is None or o[4] is not None covers all orders, making it
            #   equivalent to t < o[2]. This simplification reduces code complexity while
            #   maintaining identical behavior.
            # - Why t is not pickup time: Here, t represents the current decision epoch (0, 5, 10, ...,
            #   450), incremented by decision_interval=5 minutes, used to evaluate the system state,
            #   assign orders, and make decisions. The actual pickup time is calculated in
            #   assign_orders as pickup_time = max(o[1], t + s_p), where o[1] is r_o (ready time,
            #   t_o + 10) and s_p=4 is the pickup service time. Using t + s_p + t_travel + s_d
            #   (28 minutes) to estimate delivery completion is incorrect, as it assumes t is the
            #   pickup time, leading to premature or delayed order removal (e.g., if r_o > t + s_p,
            #   delivery occurs later). Tracking actual pickup_time (e.g., as o[5]) would allow
            #   precise removal after t < pickup_time + 28, but this adds complexity to order tuples,
            #   assign_orders, and StateManager, which is avoided for simplicity.
            # - Why this simplification: Keeping all orders until t < d_o ensures unassigned orders
            #   remain in active_orders, contributing to q_orders (count of unassigned orders with
            #   r_o <= t < d_o), critical for the state vector (s_t^7) per page 15. For assigned
            #   orders, it assumes couriers are occupied until d_o (up to 40 minutes after t_o),
            #   which overestimates occupancy (actual delivery takes 28 minutes from pickup) but
            #   simplifies state management. This reduces assignments, as couriers remain occupied
            #   longer, leaving more unassigned orders, which helps address the q_orders=0 issue
            #   observed in previous data (likely due to rapid assignments or premature order removal).
            #   The tight delivery window in assign_orders (28 minutes vs. 40-minute due time) further
            #   limits assignments, ensuring non-zero q_orders during peak periods (180–300, 360–450).
            # - Trade-off: This is less realistic for assigned orders, as couriers complete delivery in
            #   28 minutes, not 40. However, it aligns with the paper’s focus on unassigned orders for
            #   q_orders and lost orders for rewards, and avoids additional tuple fields for simplicity.
            active_orders = [o for o in active_orders if t < o[2]]

            # Compute state vector (s_t^7) at time t
            # Includes time_remaining (H - t), q_couriers (active couriers), q_orders (unassigned
            # orders with r_o <= t < d_o), and Theta1-Theta4 (state features)
            state = self.state_manager.compute_state(
                t, self.courier_scheduler, active_orders
            )
            # Select random action for exploration: (a1, a1_5) where a1, a1_5 in {0, 1, 2}
            # a1 = number of 1-hour couriers, a1_5 = number of 1.5-hour couriers
            action = random.choice(self.action_space)
            # Unpack action into number of 1-hour and 1.5-hour couriers
            a1, a1_5 = action
            # Create list of new courier types (e.g., [1, 1, 1.5] for a1=2, a1_5=1)
            new_couriers = [1] * a1 + [1.5] * a1_5
            # Calculate next epoch time (t + 5, capped at H0=450)
            t_next = min(t + self.config.decision_interval, self.config.H0)
            # Count lost orders between t and t_next: unassigned orders that become ready
            # (t <= max(r_o, t + s_p) <= t_next) but cannot be delivered by d_o

            n_lost = self.utils.get_lost_orders(
                t, t_next, active_orders, active_couriers, new_couriers, self.config
            )
            # Compute reward: K_lost * n_lost + sum(K_c[c]) (K_lost=-1, K_c={1:-0.2, 1.5:-0.25})
            # Negative reward penalizes lost orders and added couriers
            reward = self.config.K_lost * n_lost + sum(
                self.config.K_c[c] for c in new_couriers
            )

            # Add on-demand couriers starting at t + delta (5 minutes) with 1 or 1.5-hour shifts
            self.courier_scheduler.add_on_demand_couriers(t, action)
            # Add new couriers to active list: start = t + delta, end = t + delta + c * 60
            active_couriers.extend(
                [
                    (t + self.config.delta, t + self.config.delta + c * 60)
                    for c in new_couriers
                ]
            )
            # Re-assign orders with updated courier pool to account for new couriers
            active_orders, new_assignments = self.utils.assign_orders(
                t, active_orders, active_couriers, self.config
            )
            order_assignments.extend(new_assignments)

            # Update for next epoch: remove couriers whose shifts are not active at t_next
            t_next = min(t + self.config.decision_interval, self.config.H0)
            active_couriers = [
                (start, end) for start, end in active_couriers if start <= t_next < end
            ]
            # Keep orders not past due (t_next < d_o) for next epoch
            # !!! repeat updating the active order?
            active_orders = [o for o in active_orders if t_next < o[2]]
            # Compute next state (s_{t+Δ}^7) at t_next, reflecting post-action system state
            next_state = self.state_manager.compute_state(
                t_next, self.courier_scheduler, active_orders
            )
            # Store experience tuple: state (7), action (2), reward (1), next_state (7)
            data.append(state + list(action) + [reward] + next_state)

        # Return experience tuples for DQN training to optimize courier scheduling
        return data


# The reward problem, sum on the overall time interval or just one time interval?
# The threshold
# Deep learning in transportation science
