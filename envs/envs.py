
import gymnasium as gym

from data_generation.simulation import SimulationUtils, SimulationConfig
from data_generation.order_generator import OrderGenerator
from data_generation.StateManager import StateManager
from data_generation.CourierScheduler import CourierScheduler
import numpy as np



class DynamicQVRPEnv(gym.Env):
    def __init__(self, config: SimulationConfig):
        
        self.config = config
        
        self.mode = "train"
        # Initialize order generator to create orders with placement times and locations
        self.order_generator = OrderGenerator(config)
        # Initialize courier scheduler to manage base and on-demand couriers
        self.courier_scheduler = CourierScheduler(config)
        
        self.state_manager = StateManager(config)
        
        self.utils = SimulationUtils()
        
        self.active_couriers = []
        
        self.active_orders = []
        
        self.decision_epochs = np.arange(
            0, config.H0 + config.decision_interval, config.decision_interval
        )
        
    def set_mode(self, mode: str):
        assert mode in ["train", "test"], f"Unknown mode: {mode}"
        self.mode = mode
        
    
    def reset(self):

        self.active_couriers = self.courier_scheduler.base_schedule[:]
        
        if self.mode == "train":
            self.active_orders = [
                (t, r, d, loc, None, None)
                for t, r, d, loc in self.order_generator.get_orders()
            ]
        
        t = 0
        
        active_couriers = [
                (start, end) for start, end in self.active_couriers if start <= t < end
            ]
        self.active_orders = self.utils.assign_orders(
                t, self.active_orders, active_couriers, self.config
            )  # 这是临时骑手？这个为什么要比较时间，将筛选的骑手来分配订单？ 不会改变总的骑手状况吧
        
        unassigned_orders = [o for o in self.active_orders if o[4] is None]
        # Filter assigned orders that have not yet been delivered (t < delivered_time)
        assigned_but_have_not_delivered = [o for o in self.active_orders if o[4] is not None and t < o[5]]
        # Active orders are the union of unassigned and assigned-but-not-delivered orders
        active_orders = unassigned_orders + assigned_but_have_not_delivered
        
        state = self.state_manager.compute_state(
                t, self.courier_scheduler, active_orders
            )
        
        return state
    
    
    def step(self, t, action):
        a1, a1_5 = action
        new_couriers = [1]*a1 + [1.5]*a1_5
        
        # Calculate next epoch time (t + 5, capped at H0=450)
        t_next = min(t + self.config.decision_interval, self.config.H0)
        # Count lost orders between t and t_next  用更新后的骑手来计算丢失的订单
        
        # 1.更新骑手的数量，并且计算奖励
        n_lost = self.utils.get_lost_orders(
                t, t_next, self.active_orders, self.active_couriers, new_couriers, self.config
            )
        # Compute reward
        reward = self.config.K_lost * n_lost + sum(
                self.config.K_c[c] for c in new_couriers)

        # Add on-demand couriers starting at t + delta (5 minutes)
        # 2. 更新骑手的数量
        self.courier_scheduler.add_on_demand_couriers(t, action)
        
        # 3. 更新骑手的数量
        self.active_couriers.extend(
                [
                    (t + self.config.delta, t + self.config.delta + c * 60)
                    for c in new_couriers
                ]
            )
        
        # 4.用更新的骑手的数量
        self.active_orders = self.utils.assign_orders(
            t, self.active_orders, self.active_couriers, self.config
        )
        
        # Update active orders for next epoch: keep unassigned or not-yet-delivered orders not past due
        unassigned_orders = [
            o for o in self.active_orders if o[4] is None and t_next < o[2]
        ]
        assigned_but_have_not_delivered = [
            o
            for o in self.active_orders
            if o[4] is not None and t < o[5] and t_next < o[2]
        ]
        active_orders = unassigned_orders + assigned_but_have_not_delivered

        # Compute next state at t_next
        next_state = self.state_manager.compute_state(
            t_next, self.courier_scheduler, active_orders
        )
        return reward, next_state
