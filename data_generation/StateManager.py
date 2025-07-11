import numpy as np
from typing import List, Tuple
from data_generation.simulation import SimulationConfig
from data_generation.CourierScheduler import CourierScheduler


class StateManager:
    """Computes and updates the state vector (s_t^7) for the rapid delivery system."""

    def __init__(self, config: SimulationConfig):
        self.config = config

    def compute_state(
        self, t: float, courier_scheduler: CourierScheduler, active_orders: List[Tuple]
    ) -> List[float]:  # 得到的订单只是初步判断，当前时间小于订单的截止时间的所有订单都可
        """Computes the 7-dimensional state vector (s_t^7) at time t."""
        time_remaining = self.config.H - t
        q_couriers = courier_scheduler.get_active_couriers(t) 
        #返回当前时刻 t 所在的活跃骑手（骑手开始时间 ≤ t < 骑手结束时间）
        q_orders = len([o for o in active_orders if o[4] is None and o[1] <= t < o[2]])    
        #筛选当前时刻未分配且在送达时间窗口内的订单
        
        Theta1 = courier_scheduler.get_courier_changes(
            t, self.config.state_params["k1"]
        )
        Theta2 = sum(
            1
            for t_o, _, _, _, _ in active_orders
            if t - self.config.state_params["k2"] < t_o <= t
        ) 
        #筛选过去k2时刻内 新产生的订单数量
        
        
        Theta3 = sum(
            1
            for _, r_o, d_o, _, _ in active_orders
            if t < max(r_o, t + self.config.s_p) <= t + self.config.state_params["k3"]
            and max(r_o, t + self.config.s_p)
            + self.config.s_p
            + self.config.t_travel
            + self.config.s_d
            > d_o
        )
        Theta4 = (
            np.mean(
                [
                    d_o
                    - (
                        max(r_o, t + self.config.s_p)
                        + self.config.s_p
                        + self.config.t_travel
                        + self.config.s_d
                    )
                    for _, r_o, d_o, _, _ in active_orders
                    if t
                    < max(r_o, t + self.config.s_p)
                    <= t + self.config.state_params["k3"]
                    and max(r_o, t + self.config.s_p)
                    + self.config.s_p
                    + self.config.t_travel
                    + self.config.s_d
                    > d_o
                ]
            ) # 未来t3时刻可处理的订单，但是这些订单即使现在处理也要超时， 计算这些超市订单的平均延迟时间
            if Theta3 > 0
            else 0
        )
        return [time_remaining, q_couriers, q_orders, Theta1, Theta2, Theta3, Theta4]
