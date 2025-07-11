from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class SimulationConfig:
    """Stores simulation parameters from page 15 of the paper.

    This class centralizes all configuration parameters for the rapid delivery system simulation,
    following the experimental settings outlined on page 15. It includes parameters for the operating
    period, courier types, reward structure, and state vector, making it easy to modify or extend the
    simulation setup.
    """

    # Operating period in minutes (default: 540 minutes, or 9 hours)
    H: float = 540
    # Order placement period in minutes (default: 450 minutes, first 7.5 hours of H)
    H0: float = 450
    # Number of pickup locations (default: 16, as specified in the paper)
    N_pickup: int = 16
    # Show-up delay for on-demand couriers in minutes (default: 5 minutes)
    delta: float = 5
    # Decision epoch interval in minutes (default: 5 minutes, decisions every 5 minutes)
    decision_interval: float = 5
    # Courier working period lengths in hours (default: None, set to [1, 1.5] in __post_init__)
    C: List[float] = None
    # Reward (penalty) for each lost order (default: -1)
    K_lost: float = -1
    # Reward (cost) for adding couriers, mapping courier type to cost (default: None, set in __post_init__)
    K_c: Dict[float, float] = None
    # Service time at pickup location in minutes (default: 4 minutes)
    s_p: float = 4
    # Service time at dropoff location in minutes (default: 4 minutes)
    s_d: float = 4
    # Maximum number of couriers added per type per decision epoch (default: 2)
    max_couriers_per_type: int = 2
    # Range for number of 1-hour base couriers, D1 ~ U[20,30]
    D1_range: Tuple[int, int] = (20, 30)
    # Range for number of 1.5-hour base couriers, D1.5 ~ U[10,20]
    D1_5_range: Tuple[int, int] = (10, 20)
    # Parameters for state vector computation (default: None, set in __post_init__)
    state_params: Dict[str, Tuple[int, int]] = None

    def __post_init__(self):
        """Initializes default values for optional parameters.

        Sets default values for courier types (C), courier costs (K_c), and state vector
        parameters if not provided during instantiation. This ensures the simulation uses
        the paper's default settings for the 7-dimensional state vector (s_t^7).
        """
        # Set default courier types: 1-hour and 1.5-hour couriers
        self.C = [1, 1.5] if self.C is None else self.C
        # Set default courier costs: -0.2 for 1-hour, -0.25 for 1.5-hour couriers
        self.K_c = {1: -0.2, 1.5: -0.25} if self.K_c is None else self.K_c
        # Set default state parameters for s_t^7: j1,k1 for courier changes, j2,k2 for past orders,
        # j3,k3 for future late orders (j=1 for single window, k in minutes)
        self.state_params = (
            {"j1": 1, "k1": 120, "j2": 1, "k2": 30, "j3": 1, "k3": 40}
            if self.state_params is None
            else self.state_params
        )


# Utility Functions
class SimulationUtils:
    """Provides utility functions for the simulation, including order assignment and lost order calculation.

    This class contains static methods to handle common operations in the simulation, such as assigning
    orders to couriers and determining the number of orders that would be lost if no additional couriers
    are assigned. These functions are used by the Simulator class to manage system dynamics.
    """

    @staticmethod
    def assign_orders(
        t: float,
        active_orders: List[Tuple],
        active_couriers: List[Tuple],
        config: SimulationConfig,
    ) -> Tuple[List[Tuple], List[Tuple]]:
        
        
        
              
        """Assigns orders to available couriers greedily, prioritizing by due time.

        Iterates through unassigned orders and assigns them to couriers available at time t,
        ensuring the order can be delivered before its due time. Uses a greedy approach, prioritizing
        orders with earlier due times to minimize late deliveries.

        Args:
            t: Current time in minutes.
            active_orders: List of tuples (t_o, r_o, d_o, loc, assigned), where t_o is placement time,
                r_o is ready time, d_o is due time, loc is location, and assigned is courier index or None.
            active_couriers: List of tuples (start, end) representing courier shift times.
            config: SimulationConfig object with parameters like s_p (pickup time) and s_d (dropoff time).

        Returns:
            Tuple containing:
            - Updated list of orders with assignments.
            - List of new assignments as (order_idx, courier_start, courier_end).
        """
        # Identify unassigned orders that are ready (r_o <= t) and not past due (t < d_o)
        unassigned = [
            i for i, o in enumerate(active_orders) if o[4] is None and o[1] <= t < o[2]
        ]
        
        
        
        # Identify couriers available at time t (shift includes t)
        available_couriers = [
            (i, start, end)
            for i, (start, end) in enumerate(active_couriers)
            if start <= t < end
        ] # 看不懂这里的t，这里的active_couriers是啥，
        
        
        
        
        # Initialize list to store new assignments
        assignments = []
        # Create a copy of active_orders to update with assignments
        updated_orders = active_orders[:]
        # Sort unassigned orders by due time (d_o) to prioritize urgent orders
        for order_idx in sorted(unassigned, key=lambda i: active_orders[i][2]):
            order = active_orders[order_idx]
            # Calculate earliest possible pickup time (max of ready time and current time + pickup service time)
            pickup_time = max(order[1], t + config.s_p)
            # Check if order can be delivered on time (pickup_time + service times <= due time)
            if pickup_time + config.s_p + config.s_d <= order[2]:
                # Assign to the first available courier whose shift extends beyond pickup time
                for c_idx, c_start, c_end in available_couriers:
                    if pickup_time <= c_end:
                        # Update order with assigned courier index
                        updated_orders[order_idx] = (
                            order[0],
                            order[1],
                            order[2],
                            order[3],
                            c_idx,
                        )
                        # Record assignment
                        assignments.append((order_idx, c_start, c_end))
                        # Remove assigned courier from available list (each courier handles one order at a time)
                        available_couriers.remove((c_idx, c_start, c_end))
                        break
        return updated_orders, assignments

    @staticmethod
    def get_lost_orders(
        t: float,
        t_next: float,
        active_orders: List[Tuple],
        active_couriers: List[Tuple],
        config: SimulationConfig,
    ) -> int:
        """Calculates the number of orders that would be lost between t and t_next.

        Determines how many unassigned orders would miss their due time if picked up between
        t and t_next, considering both current and newly added couriers (starting at t + delta).
        An order is lost if it cannot be assigned to a courier available at its pickup time.

        Args:
            t: Current time in minutes.
            t_next: Next decision epoch time in minutes.
            active_orders: List of tuples (t_o, r_o, d_o, loc, assigned) for orders.
            active_couriers: List of tuples (start, end) for current couriers.
            new_couriers: List of courier types (e.g., [1, 1.5]) to be added at t + delta.
            config: SimulationConfig object with parameters like delta, s_p, s_d.

        Returns:
            Integer number of orders that would be lost.
        """
        
        lost = 0
        for o in active_orders:
            if o[4] is not None:
                continue
            pickup_time = max(o[1], t + config.s_p)
            if t <= pickup_time <= t_next and pickup_time + config.s_p + config.s_d > o[2]:
                can_assign = any(pickup_time <= c_end for c_start, c_end in active_couriers)
                if not can_assign:
                    lost += 1
        return lost
