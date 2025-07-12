import gymnasium as gym

from data_generation.simulation import SimulationUtils, SimulationConfig
from data_generation.order_generator import OrderGenerator
from data_generation.StateManager import StateManager
from data_generation.CourierScheduler import CourierScheduler
import numpy as np


class DynamicQVRPEnv(gym.Env):
    def __init__(self, config: SimulationConfig):

        self.config = config
        # Initialize order generator to create orders with placement times and locations
        self.order_generator = OrderGenerator(config)
        # Initialize courier scheduler to manage base and on-demand couriers
        self.courier_scheduler = CourierScheduler(config)

        self.state_manager = StateManager(config)

        self.utils = SimulationUtils()

        self.decision_epochs = np.arange(
            0, config.H0 + config.decision_interval, config.decision_interval
        )

    def reset(self):

        self.active_couriers = self.courier_scheduler.base_schedule[:]

        self.active_orders = [
            (t, r, d, loc, None) for t, r, d, loc in self.order_generator.get_orders()
        ]

    def step(self, t, action):
        a1, a1_5 = action
        new_couriers = [1] * a1 + [1.5] * a1_5
        t_next = min(t + self.config.decision_interval, self.config.H0)

        # 改变骑手
        self.courier_scheduler.add_on_demand_couriers(t, action)
        # Add new couriers to active list: start = t + delta, end = t + delta + c * 60
        self.active_couriers = self.courier_scheduler.get_active_couriers(t)
        # 计算奖励
        n_lost = self.utils.get_lost_orders(
            t, t_next, self.active_orders, self.active_couriers, self.config
        )
        reward = self.config.K_lost * n_lost + sum(
            self.config.K_c[c] for c in new_couriers
        )
        # 改变订单状态
        self.active_orders, _ = self.utils.assign_orders(
            t, self.active_orders, self.active_couriers, self.config
        )
        # 此处是否多余
        # Kang's answer: It is not redundant as the system won't place new
        # order's after H_0.
        t_next = min(t + self.config.decision_interval, self.config.H0)

        # 此处不理解，这个是临时骑手表，以便计算状态吗？
        # Kang's Answer:
        # You did not apply the active_couriers, right?
        # It can be deleted.

        active_couriers = [
            (start, end) for start, end in self.active_couriers if start <= t_next < end
        ]
        # 此处不理解，是用临时订单表，以便计算状态吗？
        # Yes, active_orders is very important for state computation.
        # However, I did not compute the state in this way,
        # as the active_orders I applied is the union of 
        # 'unassigned' and 'assigned but not delievered'
        # Check the simulator.
        active_orders = [o for o in active_orders if t_next < o[2]]
        # Compute next state (s_{t+Δ}^7) at t_next, reflecting post-action system state
        next_state = self.state_manager.compute_state(
            t_next, self.courier_scheduler, active_orders
        )
        return reward, next_state
