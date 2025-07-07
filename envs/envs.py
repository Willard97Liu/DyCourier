
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
        
        new_couriers = [1]*a1 + [1.5]*a1_5

        t_next = min(t + self.config.decision_interval, self.config.H0)
        
        
        # 丢单统计 & 奖励计算
        n_lost = self.utils.get_lost_orders(t, t_next, self.active_orders, self.active_couriers, new_couriers, self.config)
        reward = self.config.K_lost * n_lost + sum(self.config.K_c[c] for c in new_couriers)

        # 动作执行（加入新骑手）
        self.courier_scheduler.add_on_demand_couriers(t, action)
        self.active_couriers.extend([
            (t + self.config.delta, t + self.config.delta + c * 60) for c in new_couriers
        ])
        
        
        # Re-assign orders with updated courier pool
        active_orders, _ = self.utils.assign_orders(
            t, self.active_orders, self.active_couriers, self.config
        )
        
        # Compute next state at t_next
        t_next = min(t + self.config.decision_interval, self.config.H0)
        # Update active couriers for next epoch
        active_couriers = [
            (start, end) for start, end in self.active_couriers if t_next < end
        ]
        # Update active orders for next epoch
        active_orders = [
            o
            for i, o in enumerate(active_orders)
            if o[4] is None
            or (
                o[4] is not None
                and o[2] > t_next + self.config.s_p + self.config.s_d
            )
        ]
        # Compute next state (s_{t+Δ}^7)
        next_state = self.state_manager.compute_state(
            t_next, self.courier_scheduler, active_orders
        )

        return reward, next_state
