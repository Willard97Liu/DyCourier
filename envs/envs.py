import gymnasium as gym

from data_generation.simulation import SimulationUtils, SimulationConfig
from data_generation.order_generator import OrderGenerator
from data_generation.StateManager import StateManager
from data_generation.CourierScheduler import CourierScheduler
import numpy as np


class DynamicQVRPEnv(gym.Env):
    def __init__(self, config: SimulationConfig):

        self.config = config

        self.prev_lost = 0

        self.mode = "train"
        # Initialize order generator to create orders with placement times and locations

        # Initialize courier scheduler to manage base and on-demand couriers
        self.courier_scheduler = CourierScheduler(config)

        self.state_manager = StateManager(config)

        self.utils = SimulationUtils()

        self.active_couriers = []

        self.active_orders = []

        self.decision_epochs = np.arange(
            5, config.H0 + config.decision_interval, config.decision_interval
        )

    def set_mode(self, mode: str):
        assert mode in ["train", "test"], f"Unknown mode: {mode}"
        self.mode = mode

    def reset(self):

        self.active_couriers = self.courier_scheduler.base_schedule[:]

        self.prev_lost = 0

        # 给每个骑手分配唯一标识
        self.id_active_couriers = [
            (i, start, end) for i, (start, end) in enumerate(self.active_couriers)
        ]

        if self.mode == "train":
            self.order_generator = OrderGenerator(self.config)
            self.active_orders = self.order_generator.get_orders()
            # print(len(self.active_orders))

    def step(self, t, action, visible_orders):
        a1, a1_5 = action
        new_couriers = [1] * a1 + [1.5] * a1_5

        self.courier_scheduler.add_on_demand_couriers(t, action)

        new_courier = [
            (t + self.config.delta, t + self.config.delta + c * 60)
            for c in new_couriers
        ]

        self.active_couriers.extend(new_courier)

        offset = len(self.id_active_couriers)

        self.id_active_couriers.extend(
            [(offset + i, start, end) for i, (start, end) in enumerate(new_courier)]
        )

        self.utils.assign_orders(
            t,
            visible_orders,
            self.active_couriers,
            self.id_active_couriers,
            self.config,
        )

        # print(len(visible_orders))
        # 1.更新骑手的数量，并且计算奖励, 这个是计算t之前的订单，但是在t-next会不会丢失
        current_lost = self.utils.get_lost_orders(t, visible_orders, self.config)

        delta_lost = current_lost - self.prev_lost
        self.prev_lost = current_lost  # 更新为下一步的前值

        # Compute reward
        reward = self.config.K_lost * delta_lost + sum(
            self.config.K_c[c] for c in new_couriers
        )

        return reward, current_lost
