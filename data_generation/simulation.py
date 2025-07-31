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


class SimulationUtils:
    """Utility functions for simulation tasks, including order assignment and lost order calculation.

    Supports the rapid delivery system simulation by providing methods to assign orders to couriers
    and compute lost orders, aligning with the paper 'Dynamic Courier Capacity Acquisition in Rapid
    Delivery Systems: A Deep Q-Learning Approach' (page 15). Includes realistic delivery durations
    with travel time (t_travel=20) to limit assignments, helping maintain non-zero q_orders during
    peak periods (180–300, 360–450 minutes).
    """
    #To avoid circular import 
    from data_generation.order_generator import Order

    @staticmethod
    def assign_orders(
        t: float,
        visible_orders: Order,
        id_active_couriers: List[Tuple[id, float, float]],  # (id, start_time, end_time)
        config: SimulationConfig,
    ):
        # 调试
        # print(f"现在时间：{t}")
        # for o in visible_orders:
        #     print(f"[订单] 下单时间: {o.order_time:.1f}, 截止时间: {o.due_time:.1f}, 分配骑手: {o.assigned_courier}, 送达时间: {o.delivery_time}")

        unassigned_orders = [o for o in visible_orders if o.assigned_courier is None]

        available_couriers = [
            (cid, start, end)
            for (cid, start, end) in id_active_couriers
            if start <= t < end
        ]

        ## 过滤掉正在派送订单的骑手，但是还要考虑送完订单的骑手，只要分配骑手，它的delivery_time就不为空
        assigned_courier_indices = {
            o.assigned_courier
            for o in visible_orders
            if o.assigned_courier is not None
            and o.delivery_time is not None
            and t < o.delivery_time
        }

        ## 用所有骑手减去这些正在派送订单的骑手
        available_couriers = [
            (i, start, end)
            for i, start, end in available_couriers
            if i not in assigned_courier_indices
        ]
        # 调试
        # print("[可用骑手]:")
        # for cid, start, end in available_couriers:
        #     if start <= t < end:
        #         print(f"ID={cid}, 在线时间=[{start}, {end})")

        # Step 3: 给目前未分配的订单分配骑手
        # 计算这些订单的最早开始可以去取的时间和计算这个订单，从最早现在去取，并且送完后的这个时间
        # 如果这个时间小于订单的due_time, 那么这个订单可以被分配，然后找这个订单是否超过车的due_time，如果没有超过
        # 则分配这个订单，给这个订单车的编号，并且记录这个订单的理想送达时间
        for order in sorted(unassigned_orders, key=lambda o: o.due_time):
            pickup_time = max(order.ready_time, t + config.s_p)
            delivery_end_time = pickup_time + config.t_travel + config.s_d

            if (
                delivery_end_time <= order.due_time
            ):  # 这个订单理论上能够分配，就看现在有没有合适的车了
                for idx, c_start, c_end in available_couriers:
                    if delivery_end_time <= c_end:  # 有合适的车
                        # 分配成功
                        order.assigned_courier = idx
                        order.delivery_time = delivery_end_time
                        # 在可以分配的骑手中移除该骑手
                        available_couriers = [
                            c for c in available_couriers if c[0] != idx
                        ]
                        break
        # 调试
        # for o in visible_orders:
        #     print(f"[订单] 下单时间: {o.order_time:.1f}, 截止时间: {o.due_time:.1f}, 分配骑手: {o.assigned_courier}, 送达时间: {o.delivery_time}")

    @staticmethod
    def get_lost_orders(
        t: float,
        visible_orders,
        config: SimulationConfig,
    ) -> int:
        """Calculates the number of orders lost between t and t_next."""

        lost = 0

        for o in visible_orders:
            if o.assigned_courier is not None:
                continue

            if t < config.H0:
                # 不是最后一刻，只判断当前出发也送不完的订单
                pickup_time = max(o.ready_time, t + config.s_p)
                delivery_end_time = pickup_time + config.t_travel + config.s_d
                if delivery_end_time > o.due_time:
                    lost += 1

            else:
                # 是最后一刻，所有还没被分配的订单都算 lost
                lost += 1

        return lost
