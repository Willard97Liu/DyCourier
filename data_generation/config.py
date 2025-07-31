from dataclasses import dataclass
from typing import List, Tuple, Dict

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
    N_pickup: int = 16
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
    state_params: Dict[str, Tuple[int, int]] = None

    def __post_init__(self):
        """Initializes default values for optional parameters after instantiation.

        Sets default values for courier types (C), courier costs (K_c), and state parameters
        if not provided during instantiation. Ensures the simulation uses consistent settings
        as specified in the paper (page 15) for reproducibility and alignment with the MDP.
        """
        # Set default courier types if not provided: 1-hour and 1.5-hour shifts
        # Matches the paper’s specification of two courier types
        self.C = [1, 1.5] if self.C is None else self.C
        # Set default courier costs if not provided: -0.2 for 1-hour, -0.25 for 1.5-hour
        # Negative values penalize adding couriers in the reward function
        self.K_c = {1: -0.2, 1.5: -0.25} if self.K_c is None else self.K_c
        # Set default state parameters if not provided
        # j1, j2, j3: weights (set to 1); k1, k2, k3: time windows (120, 30, 40 minutes)
        # Used for computing Theta1 (courier changes), Theta2 (recent orders), Theta3/Theta4 (late orders)
        self.state_params = (
            {"j1": 1, "k1": 120, "j2": 1, "k2": 30, "j3": 1, "k3": 40}
            if self.state_params is None
            else self.state_params
        )


