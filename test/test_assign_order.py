import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from data_generation.simulation import SimulationConfig, SimulationUtils
from data_generation.StateManager import StateManager
from data_generation.simulator import Simulator

config = SimulationConfig()


# Tests for SimulationUtils
def test_assign_orders_no_orders():
    t = 0
    config = SimulationConfig()
    active_orders = []
    active_couriers = [(0, 60), (10, 70)]
    result = SimulationUtils.assign_orders(t, active_orders, active_couriers, config)
    assert result == [], "Should return empty list when no orders"


def test_assign_orders_no_couriers():
    t = 0
    config = SimulationConfig()
    active_orders = [(0, 10, 40, 0, None, None), (5, 15, 45, 1, None, None)]
    active_couriers = []
    result = SimulationUtils.assign_orders(t, active_orders, active_couriers, config)
    assert all(
        o[4] is None for o in result
    ), "No assignments should be made with no couriers"
    assert all(o[5] is None for o in result), "delivered_time should remain None"


def test_assign_orders_successful_assignment():
    t = 0
    config = SimulationConfig()
    active_orders = [(0, 10, 40, 0, None, None)]
    active_couriers = [(0, 60)]  # Courier available for entire delivery
    result = SimulationUtils.assign_orders(t, active_orders, active_couriers, config)
    assert len(result) == 1, "Should return one order"
    assert result[0][4] is not None, "Order should be assigned"
    assert (
        result[0][5] == 34
    ), "delivered_time should be pickup_time (10) + t_travel (20) + s_d (4)"


def test_assign_orders_past_due():
    t = 50
    config = SimulationConfig()
    active_orders = [(0, 10, 40, 0, None, None)]  # Past due (t >= d_o)
    active_couriers = [(0, 60)]
    result = SimulationUtils.assign_orders(t, active_orders, active_couriers, config)
    assert result[0][4] is None, "Past-due order should not be assigned"
    assert result[0][5] is None, "delivered_time should remain None"
