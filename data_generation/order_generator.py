import numpy as np
from typing import List
from data_generation.simulation import SimulationConfig


class Order:
    def __init__(self, t, r, d, loc):
        self.order_time = t            # 下单时间
        self.ready_time = r            # 准备时间
        self.due_time = d              # 截止时间
        self.location = loc            # 接单地点
        self.assigned_courier = None   # 已分配骑手 ID
        self.delivery_time = None      # 实际送达时间
        self.status = "unassigned"     # 状态：unassigned / assigned / delivered / lost

    def is_visible(self, t):
        return self.order_time <= t

    def is_active(self, t):
        return self.ready_time <= t < self.due_time and self.status != "delivered"

    def is_lost(self, t):
        return t >= self.due_time and self.status != "delivered"


class OrderGenerator:
    """Generates orders with peaks during lunch and dinner times."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.orders_per_location = np.random.randint(10, 21, config.N_pickup)
        self.total_orders = sum(self.orders_per_location)
        self.order_times = self._generate_order_times()
        self.order_ready_times = self.order_times + 10
        self.order_due_times = self.order_times + 40
        self.order_locations = self._generate_locations()

    def _generate_order_times(self) -> np.ndarray:
        n_uniform = self.total_orders // 2
        n_lunch = self.total_orders // 4
        n_dinner = self.total_orders - n_uniform - n_lunch

        uniform_times = np.random.uniform(0, self.config.H0, n_uniform)
        lunch_times = np.random.normal(loc=240, scale=30, size=n_lunch)
        dinner_times = np.random.normal(loc=405, scale=30, size=n_dinner)

        order_times = np.concatenate([uniform_times, lunch_times, dinner_times])
        order_times = np.clip(order_times, 0, self.config.H0)
        return np.sort(order_times)

    def _generate_locations(self) -> List[int]:
        locations = []
        for i, n_d in enumerate(self.orders_per_location):
            locations.extend([i] * n_d)
        locations = np.array(locations)
        order_data = list(
            zip(
                self.order_times,
                self.order_ready_times,
                self.order_due_times,
                locations,
            )
        )
        np.random.shuffle(order_data)
        _, _, _, locations = zip(*order_data)
        return list(locations)

    def get_orders(self) -> List[Order]:
        return [
            Order(t, r, d, loc)
            for t, r, d, loc in zip(
                self.order_times,
                self.order_ready_times,
                self.order_due_times,
                self.order_locations,
            )
        ]
