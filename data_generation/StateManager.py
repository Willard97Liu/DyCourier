import numpy as np
from typing import List, Tuple
from data_generation.config import SimulationConfig
from data_generation.order_generator import Order
from data_generation.CourierScheduler import CourierScheduler


class StateManager:
    """Computes and updates the state vector (s_t^7) for the rapid delivery system."""

    def __init__(self, config: SimulationConfig):
        self.config = config

    def compute_state(
        self, t: float, courier_scheduler: CourierScheduler, visible_orders: List[Order]
    ) -> List[float]:
        """Computes the 7-dimensional state vector (s_t^7) at time t."""
        time_remaining = self.config.H - t
        q_couriers = courier_scheduler.get_active_couriers(t)

        # 未被分配，且当前时间在 ready_time 和 due_time 之间的订单数量
        q_orders = len(
            [
                o
                for o in visible_orders
                if o.assigned_courier is None and o.ready_time <= t < o.due_time
            ]
        )

        Theta1 = courier_scheduler.get_courier_changes(
            t, self.config.state_params["k1"]
        )

        # 最近 k2 时间内产生的订单数
        Theta2 = sum(
            1
            for o in visible_orders
            if t - self.config.state_params["k2"] < o.order_time <= t
        )

        # Theta3: 当前时间之后将在 k3 时间窗口内 ready，但无法按时送达的订单数量
        Theta3 = sum(
            1
            for o in visible_orders
            if t
            < max(o.ready_time, t + self.config.s_p)
            <= t + self.config.state_params["k3"]
            and max(o.ready_time, t + self.config.s_p)
            + self.config.t_travel
            + self.config.s_d
            > o.due_time
        )

        # Theta4: 上述订单的平均违约时间（只对 Theta3 > 0 的情况）
        Theta4 = (
            np.mean(
                [
                    o.due_time
                    - (
                        max(o.ready_time, t + self.config.s_p)
                        + self.config.s_p
                        + self.config.t_travel
                        + self.config.s_d
                    )
                    for o in visible_orders
                    if t
                    < max(o.ready_time, t + self.config.s_p)
                    <= t + self.config.state_params["k3"]
                    and max(o.ready_time, t + self.config.s_p)
                    + self.config.t_travel
                    + self.config.s_d
                    > o.due_time
                ]
            )
            if Theta3 > 0
            else 0
        )

        return [time_remaining, q_couriers, q_orders, Theta1, Theta2, Theta3, Theta4]
