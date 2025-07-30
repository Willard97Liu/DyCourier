from dataclasses import dataclass
from typing import List, Tuple, Dict
from dataclasses import field
from collections import defaultdict


@dataclass
class SimulationConfig:
    """Stores simulation parameters based on page 15 of the paper 'Dynamic Courier Capacity Acquisition in Rapid Delivery Systems: A Deep Q-Learning Approach.'

    This class centralizes all configuration parameters required for the rapid delivery system simulation. It defines the temporal, spatial, and operational settings, including service and travel times, courier types, and state vector parameters. The settings are derived from the paper’s experimental setup (page 15), with an added travel time parameter to model realistic delivery durations between pickup and dropoff locations. The configuration ensures consistency across simulation components (Simulator, OrderGenerator, CourierScheduler, StateManager) and supports the Markov Decision Process (MDP) for DQN training.
    """

    # Total operating period in minutes (H = 540 minutes, or 9 hours, from 9:00 AM to 6:00 PM)
    # Represents the full simulation duration, during which orders are placed and delivered
    H: float = 540
    # Order placement period in minutes (H0 = 450 minutes, or 7.5 hours, from 9:00 AM to 4:30 PM)
    # Orders can only be placed within [0, H0], after which only deliveries continue until H
    H0: float = 450
    # Number of pickup locations (N_pickup = 16, representing restaurants or stores)
    # Each location generates orders independently, as specified in the paper
    N_pickup: int = 3
    # Show-up delay for on-demand couriers in minutes (delta = 5)
    # On-demand couriers added at time t start their shift at t + delta
    delta: float = 5
    # Decision epoch interval in minutes (decision_interval = 5)
    # The simulation makes decisions (e.g., add couriers) every 5 minutes, as per page 15
    decision_interval: float = 5
    # List of courier types (C = [1, 1.5] hours, representing shift durations of 1 hour and 1.5 hours)
    # None by default, initialized in __post_init__ to align with the paper’s settings
    C: List[float] = None
    # Penalty for each lost order (K_lost = -1)
    # Negative value penalizes lost orders in the reward function (r_t = K_lost * n_lost + courier costs)
    K_lost: float = -1
    # Cost of adding couriers (K_c = {1: -0.2, 1.5: -0.25} for 1-hour and 1.5-hour couriers)
    # None by default, initialized in __post_init__; negative costs reflect penalties in the reward
    K_c: Dict[float, float] = None
    # Pickup service time in minutes (s_p = 4)
    # Time spent at the pickup location (e.g., collecting the order from a restaurant)
    s_p: float = 4
    # Dropoff service time in minutes (s_d = 4)
    # Time spent at the dropoff location (e.g., handing the order to the customer)
    s_d: float = 4
    # Travel time between pickup and dropoff locations in minutes (t_travel = 20)
    # Represents average urban delivery travel time; not in the paper but added for realism
    t_travel: float = 20
    # Maximum number of couriers added per type per decision epoch (max_couriers_per_type = 2)
    # Limits actions to adding 0, 1, or 2 couriers of each type (1-hour, 1.5-hour) per epoch
    max_couriers_per_type: int = 2
    # Range for number of base 1-hour couriers (D1 ~ Uniform[20,30])
    # Defines the initial number of 1-hour couriers scheduled at the start of the simulation
    D1_range: Tuple[int, int] = (20, 30)
    # Range for number of base 1.5-hour couriers (D1.5 ~ Uniform[10,20])
    # Defines the initial number of 1.5-hour couriers scheduled at the start
    D1_5_range: Tuple[int, int] = (10, 20)
    # State vector parameters for computing Theta1, Theta2, Theta3, Theta4
    # Defines time windows and weights for state components (e.g., k1 = 120 minutes for courier changes)
    # None by default, initialized in __post_init__
    state_params: Dict[str, int] = field(
        default_factory=lambda: {
            "k1": 10,
            "k2": 15,
            "k3": 20,
        }
    )

    def __post_init__(self):
        """Initializes default values for optional parameters after instantiation."""

        # Set default courier types
        if self.C is None:
            self.C = [1, 1.5]

        # Set default courier costs
        if self.K_c is None:
            self.K_c = {1: -0.2, 1.5: -0.25}

        # Ensure all expected keys exist in state_params
        default_state_params = {
            "j1": 1,
            "k1": 120,
            "j2": 1,
            "k2": 30,
            "j3": 1,
            "k3": 40,
        }

        for key, value in default_state_params.items():
            self.state_params.setdefault(key, value)


class SimulationUtils:
    """Utility functions for simulation tasks, including order assignment and lost order calculation.

    Supports the rapid delivery system simulation by providing methods to assign orders to couriers
    and compute lost orders, aligning with the paper 'Dynamic Courier Capacity Acquisition in Rapid
    Delivery Systems: A Deep Q-Learning Approach' (page 15). Includes realistic delivery durations
    with travel time (t_travel=20) to limit assignments, helping maintain non-zero q_orders during
    peak periods (180–300, 360–450 minutes).
    """

    @staticmethod
    def assign_orders(
        t: float,
        active_orders: List[Tuple],
        active_couriers: List[Tuple],
        config: SimulationConfig,
    ) -> List[Tuple]:
        """Assigns orders to available couriers greedily, prioritizing by due time.

        Iterates through unassigned orders placed by time t (t_o <= t) and not past due (t < d_o).
        Assigns each order to an available courier whose shift covers the entire delivery process
        (pickup, travel, and dropoff). Prioritizes orders by due time to minimize late deliveries,
        following the greedy assignment strategy implied in the paper (page 15). Ensures each courier
        handles at most one order at a time, aligning with the simulation’s one-order-per-courier
        assumption. The inclusion of travel time (t_travel = 20 minutes) extends the delivery duration
        to 28 minutes (s_p + t_travel + s_d), reducing assignments and increasing unassigned orders
        (q_orders) during peak periods, addressing the q_orders=0 issue observed in previous data.

        Args:
            t: Current time in minutes (decision epoch, e.g., t = 0, 5, 10, ..., 450).
            active_orders: List of tuples representing orders, each with:
                - t_o: Order placement time (minutes, t_o in [0, H0]).
                - r_o: Order ready time (t_o + 10 minutes, when order is available for pickup).
                - d_o: Order due time (t_o + 40 minutes, latest delivery time).
                - loc: Pickup location index (integer, 0 to N_pickup-1).
                - assigned: Courier index (integer) or None if unassigned.
                - delivered_time: Delivery completion time (minutes) or None if not delivered.
            active_couriers: List of tuples representing courier shifts, each with:
                - start: Shift start time (minutes).
                - end: Shift end time (minutes, start + 60 or 90 for 1-hour or 1.5-hour couriers).
            config: SimulationConfig object containing parameters:
                - s_p: Pickup service time (4 minutes).
                - s_d: Dropoff service time (4 minutes).
                - t_travel: Travel time (20 minutes).
                - Other parameters (e.g., H0, delta) for simulation consistency.

        Returns:
            Updated list of orders with new assignments (same tuple structure).
        """
        # Find indices of unassigned orders placed by time t (t_o <= t) and not past due (t < d_o)

        # Compute busy-until time per courier using hashing map
        courier_busy_until = defaultdict(lambda: 0)
        for o in active_orders:
            assigned, delivered = o[4], o[5]
            if assigned is not None and delivered is not None:
                courier_busy_until[assigned] = max(
                    courier_busy_until[assigned], delivered
                )

        # Available couriers: shift is active and not currently busy
        available_couriers = [
            (i, start, end)
            for (i, start, end) in active_couriers
            if start <= t < end and courier_busy_until[i] <= t
        ]

        # Copy active_orders to update assignments
        updated_orders = active_orders[:]

        # Sort unassigned and not-past-due orders by due time
        unassigned = [
            i for i, o in enumerate(active_orders) if o[4] is None and o[0] <= t < o[2]
        ]

        for order_idx in sorted(unassigned, key=lambda i: active_orders[i][2]):
            order = active_orders[order_idx]
            pickup_time = max(order[1], t + config.s_p)
            delivery_time = pickup_time + config.t_travel + config.s_d

            if delivery_time > order[2]:
                continue  # too late to deliver

            for c_idx, c_start, c_end in available_couriers:
                if delivery_time <= c_end:
                    updated_orders[order_idx] = (
                        order[0],  # t_o
                        order[1],  # r_o
                        order[2],  # d_o
                        order[3],  # loc
                        c_idx,  # assigned courier
                        delivery_time,
                    )
                    # Mark this courier as busy
                    courier_busy_until[c_idx] = delivery_time
                    # Remove from availability
                    available_couriers = [
                        (i, s, e) for (i, s, e) in available_couriers if i != c_idx
                    ]
                    break

        return updated_orders

    @staticmethod
    def get_lost_orders(
        t: float,
        t_next: float,
        active_orders: List[Tuple],
        active_couriers: List[Tuple],
        new_couriers: List[float],
        config: SimulationConfig,
    ) -> int:
        """Calculates the number of orders lost between t and t_next.

        Identifies unassigned orders that become ready between t and t_next (t <= pickup_time <= t_next)
        but cannot be delivered by their due time due to insufficient courier availability or time
        constraints. A lost order contributes to the reward penalty (K_lost * n_lost) in the MDP.
        Incorporates travel time (t_travel = 20 minutes) for realistic delivery duration, aligning
        with the paper’s lost order definition (page 15). The longer delivery duration may increase
        lost orders, helping retain unassigned orders in active_orders, which increases q_orders.

        Args:
            t: Current time in minutes (start of the current decision epoch).
            t_next: Next decision epoch time (t + decision_interval, or H0 if at the end).
            active_orders: List of tuples (t_o, r_o, d_o, loc, assigned), same structure as assign_orders.
            active_couriers: List of tuples (start, end) for current courier shifts.
            new_couriers: List of courier types (1 or 1.5 hours) to be added at t + delta.
            config: SimulationConfig object with parameters:
                - s_p: Pickup service time (4 minutes).
                - s_d: Dropoff service time (4 minutes).
                - t_travel: Travel time (20 minutes).
                - delta: Show-up delay for new couriers (5 minutes).

        Returns:
            Integer number of lost orders, used to compute the reward penalty (K_lost * n_lost).
        """
        # Initialize counter for lost orders
        lost = 0
        # Create a temporary list of couriers, including current active couriers
        # and new couriers starting at t + delta with their respective shift durations
        temp_couriers = active_couriers + [
            (t + config.delta, t + config.delta + c * 60) for c in new_couriers
        ]
        # Iterate through all orders to check for potential losses
        for i, o in enumerate(active_orders):
            # Skip orders that are already assigned (not eligible for loss)
            if o[4] is not None:
                continue
            # Assign courier
            # Calculate the earliest possible pickup time for the order
            # max(r_o, t + s_p) accounts for the order’s ready time and courier travel to pickup
            pickup_time = max(o[1], t + config.s_p)
            # Check if the order becomes ready in the interval [t, t_next]
            # and cannot be delivered by its due time (d_o)
            # Delivery requires s_p + t_travel + s_d = 28 minutes
            # Check the inequality
            if (
                t <= pickup_time <= t_next
                and pickup_time + config.t_travel + config.s_d > o[2]
            ):
                # Initialize flag to check if the order can be assigned to any courier
                can_assign = False
                # Check if any courier (current or new) has a shift that covers the delivery
                for c_start, c_end in temp_couriers:
                    # Ensure the courier’s shift extends to cover pickup + travel + dropoff
                    if pickup_time + config.t_travel + config.s_d <= c_end:
                        # If a courier is available, the order can be assigned
                        can_assign = True
                        # Break to avoid checking further couriers
                        break
                # If no courier can deliver the order on time, increment the lost counter
                if not can_assign:

                    lost += 1
        # Return the total number of lost orders for the reward calculation
        return lost
