
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
        
        # Calculate next epoch time (t + 5, capped at H0=450)
        t_next = min(t + self.config.decision_interval, self.config.H0)
        # Count lost orders between t and t_next
        n_lost = self.utils.get_lost_orders(
                t, t_next, self.active_orders, self.active_couriers, new_couriers, self.config
            )
        # Compute reward
        reward = self.config.K_lost * n_lost + sum(
                self.config.K_c[c] for c in new_couriers)

        # Add on-demand couriers starting at t + delta (5 minutes)
        self.courier_scheduler.add_on_demand_couriers(t, action)
        # Add new couriers to active list
        self.active_couriers.extend(
                [
                    (t + self.config.delta, t + self.config.delta + c * 60)
                    for c in new_couriers
                ]
            )
        # Re-assign orders with updated courier pool
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
