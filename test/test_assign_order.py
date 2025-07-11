import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from unittest.mock import Mock, patch
from data_generation.simulation import SimulationConfig, SimulationUtils
from data_generation.StateManager import StateManager
from data_generation.simulator import Simulator

# Test fixtures


@pytest.fixture
def config():
    return SimulationConfig()


@pytest.fixture
def courier_scheduler():
    scheduler = Mock()
    scheduler.base_schedule = [
        (0, 60),
        (10, 70),
        (20, 110),
    ]  # 1-hour and 1.5-hour shifts
    scheduler.add_on_demand_couriers = Mock()
    return scheduler


@pytest.fixture
def state_manager(config):
    return StateManager(config)


@pytest.fixture
def utils():
    return SimulationUtils()


@pytest.fixture
def decision_epochs():
    return [0, 5, 10, 15]


@pytest.fixture
def action_space():
    return [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]


@pytest.fixture
def simulator(
    config,
    courier_scheduler,
    order_generator,
    state_manager,
    utils,
    decision_epochs,
    action_space,
):
    return Simulator(
        config,
        courier_scheduler,
        order_generator,
        state_manager,
        utils,
        decision_epochs,
        action_space,
    )


# Tests for SimulationUtils
def test_assign_orders_no_orders(config, utils):
    t = 0
    active_orders = []
    active_couriers = [(0, 60), (10, 70)]
    result = utils.assign_orders(t, active_orders, active_couriers, config)
    assert result == [], "Should return empty list when no orders"


def test_assign_orders_no_couriers(config, utils):
    t = 0
    active_orders = [(0, 10, 40, 0, None, None), (5, 15, 45, 1, None, None)]
    active_couriers = []
    result = utils.assign_orders(t, active_orders, active_couriers, config)
    assert all(
        o[4] is None for o in result
    ), "No assignments should be made with no couriers"
    assert all(o[5] is None for o in result), "delivered_time should remain None"


def test_assign_orders_successful_assignment(config, utils):
    t = 0
    active_orders = [(0, 10, 40, 0, None, None)]
    active_couriers = [(0, 60)]  # Courier available for entire delivery
    result = utils.assign_orders(t, active_orders, active_couriers, config)
    assert len(result) == 1, "Should return one order"
    assert result[0][4] is not None, "Order should be assigned"
    assert (
        result[0][5] == 34
    ), "delivered_time should be pickup_time (10) + t_travel (20) + s_d (4)"


def test_assign_orders_past_due(config, utils):
    t = 50
    active_orders = [(0, 10, 40, 0, None, None)]  # Past due (t >= d_o)
    active_couriers = [(0, 60)]
    result = utils.assign_orders(t, active_orders, active_couriers, config)
    assert result[0][4] is None, "Past-due order should not be assigned"
    assert result[0][5] is None, "delivered_time should remain None"
