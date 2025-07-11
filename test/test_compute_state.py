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
    scheduler.get_courier_changes = Mock(return_value=0)  # Mock for Theta1
    return scheduler


@pytest.fixture
def order_generator():
    generator = Mock()
    generator.get_orders.return_value = [
        (0, 10, 40, 0),  # t_o, r_o, d_o, loc
        (5, 15, 45, 1),
        (10, 20, 50, 2),
    ]
    return generator


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
    active_couriers = [(0, 60)]
    result = utils.assign_orders(t, active_orders, active_couriers, config)
    assert len(result) == 1, "Should return one order"
    assert result[0][4] is not None, "Order should be assigned"
    assert (
        result[0][5] == 34
    ), "delivered_time should be pickup_time (10) + t_travel (20) + s_d (4)"


def test_assign_orders_past_due(config, utils):
    t = 50
    active_orders = [(0, 10, 40, 0, None, None)]
    active_couriers = [(0, 60)]
    result = utils.assign_orders(t, active_orders, active_couriers, config)
    assert result[0][4] is None, "Past-due order should not be assigned"
    assert result[0][5] is None, "delivered_time should remain None"


# Tests for StateManager
def test_compute_state_no_orders(config, courier_scheduler, state_manager):
    t = 0
    active_orders = []
    state = state_manager.compute_state(t, courier_scheduler, active_orders)
    assert len(state) == 7, "State should have 7 elements"
    assert state[0] == 540, "time_remaining should be H (540)"
    assert state[1] == 3, "q_couriers should match active couriers"
    assert state[2] == 0, "q_orders should be 0"
    assert state[3] == 0, "Theta1 should be 0 (mocked courier changes)"
    assert state[4:7] == [0, 0, 0], "Theta2-4 should be 0 with no orders"


def test_compute_state_with_orders(config, courier_scheduler, state_manager):
    t = 0
    active_orders = [(0, 10, 40, 0, None, None), (5, 15, 45, 1, None, None)]
    state = state_manager.compute_state(t, courier_scheduler, active_orders)
    assert state[0] == 540, "time_remaining should be H - t"
    assert state[1] == 3, "q_couriers should match active couriers"
    assert state[2] == 2, "q_orders should be 2 (unassigned, not past due)"
    assert state[3] == 0, "Theta1 should be 0 (mocked courier changes)"
    assert state[4] == 0.5, "Theta2: 1 order at loc 0 / 2 orders"
    assert state[5] == 0.5, "Theta3: 1 order at loc 1 / 2 orders"
    assert state[6] == 0, "Theta4: no orders at loc 2"


def test_compute_state_past_due_orders(config, courier_scheduler, state_manager):
    t = 50
    active_orders = [(0, 10, 40, 0, None, None)]
    state = state_manager.compute_state(t, courier_scheduler, active_orders)
    assert state[2] == 0, "q_orders should be 0 (past due)"
    assert state[3] == 0, "Theta1 should be 0 (mocked courier changes)"
    assert state[4:7] == [0, 0, 0], "Theta2-4 should be 0 with no valid orders"


# Tests for Simulator
@patch("random.choice")
def test_run_episode_empty_episode(
    mock_random_choice, simulator, config, courier_scheduler, order_generator
):
    order_generator.get_orders.return_value = []
    mock_random_choice.return_value = (0, 0)
    courier_scheduler.base_schedule = []
    result = simulator.run_episode()
    assert len(result) == len(simulator.decision_epochs), "One tuple per epoch"
    for tuple_data in result:
        assert len(tuple_data) == 17, "Each tuple should have 7+2+1+7 elements"
        assert (
            tuple_data[0]
            == config.H - simulator.decision_epochs[result.index(tuple_data)]
        ), "time_remaining correct"
        assert tuple_data[1] == 0, "q_couriers should be 0"
        assert tuple_data[2] == 0, "q_orders should be 0"
        assert tuple_data[3] == 0, "Theta1 should be 0"
        assert tuple_data[4:7] == [0, 0, 0], "Theta2-4 should be 0"
        assert tuple_data[7:9] == [0, 0], "Action should be (0, 0)"
        assert tuple_data[9] == 0, "Reward should be 0 (no lost orders, no couriers)"


@patch("random.choice")
def test_run_episode_with_orders_and_couriers(mock_random_choice, simulator, config):
    mock_random_choice.return_value = (1, 0)
    simulator.utils.get_lost_orders = Mock(return_value=1)
    result = simulator.run_episode()
    assert len(result) == len(simulator.decision_epochs), "One tuple per epoch"
    for tuple_data in result:
        assert len(tuple_data) == 17, "Each tuple should have 7+2+1+7 elements"
        t = simulator.decision_epochs[result.index(tuple_data)]
        assert tuple_data[0] == config.H - t, "time_remaining correct"
        assert tuple_data[7:9] == [1, 0], "Action should be (1, 0)"
        assert (
            tuple_data[9] == config.K_lost * 1 + config.K_c[1]
        ), "Reward includes lost order and courier cost"


def test_run_episode_delivered_time_handling(simulator, config):
    simulator.order_generator.get_orders.return_value = [(0, 10, 40, 0)]
    simulator.courier_scheduler.base_schedule = [(0, 60)]
    result = simulator.run_episode()
    for tuple_data in result:
        t = simulator.decision_epochs[result.index(tuple_data)]
        if t == 0:
            assert any(
                o[4] is not None and o[5] == 34
                for o in simulator.utils.assign_orders(
                    t, [(0, 10, 40, 0, None, None)], [(0, 60)], config
                )
            ), "Order should be assigned with delivered_time=34"
        if t >= 34:
            assert (
                simulator.state_manager.compute_state(
                    t, simulator.courier_scheduler, [(0, 10, 40, 0, 0, 34)]
                )[2]
                == 0
            ), "No active orders after delivery"
