import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from unittest.mock import Mock, patch
from data_generation.simulation import SimulationConfig, SimulationUtils
from data_generation.StateManager import StateManager
from data_generation.simulator import Simulator
from data_generation.CourierScheduler import CourierScheduler
from data_generation.order_generator import OrderGenerator


# Tests for SimulationUtils
def test_assign_orders_no_orders():
    t = 0
    active_orders = []
    active_couriers = [(0, 60), (10, 70)]
    config = SimulationConfig()
    result = SimulationUtils.assign_orders(t, active_orders, active_couriers, config)
    assert result == [], "Should return empty list when no orders"


def test_assign_orders_no_couriers():
    t = 0
    active_orders = [(0, 10, 40, 0, None, None), (5, 15, 45, 1, None, None)]
    active_couriers = []
    config = SimulationConfig()
    result = SimulationUtils.assign_orders(t, active_orders, active_couriers, config)
    assert all(
        o[4] is None for o in result
    ), "No assignments should be made with no couriers"
    assert all(o[5] is None for o in result), "delivered_time should remain None"


def test_assign_orders_successful_assignment():
    t = 0
    active_orders = [(0, 10, 40, 0, None, None)]
    active_couriers = [(0, 60)]
    config = SimulationConfig()
    result = SimulationUtils.assign_orders(t, active_orders, active_couriers, config)
    assert len(result) == 1, "Should return one order"
    assert result[0][4] is not None, "Order should be assigned"
    assert (
        result[0][5] == 34
    ), "delivered_time should be pickup_time (10) + t_travel (20) + s_d (4)"


def test_assign_orders_past_due():
    t = 50
    active_orders = [(0, 10, 40, 0, None, None)]
    active_couriers = [(0, 60)]
    config = SimulationConfig()
    result = SimulationUtils.assign_orders(t, active_orders, active_couriers, config)
    assert result[0][4] is None, "Past-due order should not be assigned"
    assert result[0][5] is None, "delivered_time should remain None"


# Tests for StateManager
def test_compute_state_no_orders():
    t = 0
    active_orders = []
    config = SimulationConfig()
    state_manager = StateManager(config)
    courier_scheduler = CourierScheduler(config)
    state = state_manager.compute_state(t, courier_scheduler, active_orders)
    assert len(state) == 7
    assert state[0] == 540

    assert state[1] == 3

    assert state[2] == 0
    assert state[3] == 0
    assert state[4:7] == [0, 0, 0]


def test_compute_state_with_orders():
    t = 0
    active_orders = [(0, 10, 40, 0, None, None), (5, 15, 45, 1, None, None)]
    config = SimulationConfig()
    state_manager = StateManager(config)
    courier_scheduler = CourierScheduler(config)
    state = state_manager.compute_state(t, courier_scheduler, active_orders)
    assert state[0] == 540
    assert state[1] == 3
    assert state[2] == 2
    assert state[3] == 0
    assert state[4] == 0.5
    assert state[5] == 0.5
    assert state[6] == 0


def test_compute_state_past_due_orders():
    t = 50
    active_orders = [(0, 10, 40, 0, None, None)]
    config = SimulationConfig()
    state_manager = StateManager(config)
    courier_scheduler = CourierScheduler(config)
    state = state_manager.compute_state(t, courier_scheduler, active_orders)
    assert state[2] == 0
    assert state[3] == 0
    assert state[4:7] == [0, 0, 0]


# Tests for Simulator
@patch("random.choice")
def test_run_episode_empty_episode(mock_random_choice):
    config = SimulationConfig()
    OrderGenerator.get_orders.return_value = []
    mock_random_choice.return_value = (0, 0)
    CourierScheduler.base_schedule = []
    sim = Simulator(config)
    result = sim.run_episode()
    for idx, tuple_data in enumerate(result):
        t = sim.decision_epochs[idx]
        assert len(tuple_data) == 17
        assert tuple_data[0] == config.H - t
        assert tuple_data[1] == 0
        assert tuple_data[2] == 0
        assert tuple_data[3] == 0
        assert tuple_data[4:7] == [0, 0, 0]
        assert tuple_data[7:9] == [0, 0]
        assert tuple_data[9] == 0


@patch("random.choice")
def test_run_episode_with_orders_and_couriers(mock_random_choice):
    config = SimulationConfig()
    mock_random_choice.return_value = (1, 0)
    sim = Simulator(config)
    SimulationUtils.get_lost_orders = Mock(return_value=1)
    result = sim.run_episode()
    for idx, tuple_data in enumerate(result):
        t = sim.decision_epochs[idx]
        assert len(tuple_data) == 17
        assert tuple_data[0] == config.H - t
        assert tuple_data[7:9] == [1, 0]
        assert tuple_data[9] == config.K_lost * 1 + config.K_c[1]


def test_run_episode_delivered_time_handling():
    config = SimulationConfig()
    OrderGenerator.get_orders.return_value = [(0, 10, 40, 0)]
    state_manager = StateManager(config)
    courier_scheduler = CourierScheduler(config)
    courier_scheduler.base_schedule = [(0, 60)]
    sim = Simulator(config)
    result = sim.run_episode()
    for idx, tuple_data in enumerate(result):
        t = sim.decision_epochs[idx]
        if t == 0:
            assigned_orders = SimulationUtils.assign_orders(
                t, [(0, 10, 40, 0, None, None)], [(0, 60)], config
            )
            assert any(
                o[4] is not None and o[5] == 34 for o in assigned_orders
            ), "Order should be assigned with delivered_time=34"
        if t >= 34:
            state = state_manager.compute_state(
                t, courier_scheduler, [(0, 10, 40, 0, 0, 34)]
            )
            assert state[2] == 0, "No active orders after delivery"
