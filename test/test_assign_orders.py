import sys
import os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from data_generation.Config import SimulationConfig
from data_generation.Utils import SimulationUtils
from data_generation.OrderGenerator import Order

def test_assign_orders_no_orders():
    t = 0
    config = SimulationConfig()
    visible_orders = []
    active_couriers = [(0, 0, 60), (1, 10, 70)]

    SimulationUtils.assign_orders(t, visible_orders, active_couriers, config)

    assert visible_orders == [], "No orders should remain no orders"


def test_assign_orders_no_couriers():
    t = 0
    config = SimulationConfig()
    orders = [
        Order(t=0, r=10, d=40, loc=0),
        Order(t=5, r=15, d=45, loc=1),
    ]
    active_couriers = []

    SimulationUtils.assign_orders(t, orders, active_couriers, config)

    for o in orders:
        assert o.assigned_courier is None
        assert o.delivery_time is None
        assert o.status == "unassigned"


def test_assign_orders_successful_assignment():
    t = 0
    config = SimulationConfig()
    order = Order(t=0, r=10, d=40, loc=0)
    orders = [order]
    active_couriers = [(0, 0, 60)]  # Courier is active the entire time

    SimulationUtils.assign_orders(t, orders, active_couriers, config)

    assert order.assigned_courier == 0
    expected_delivery = 10 + config.t_travel + config.s_d
    assert order.delivery_time == expected_delivery
    assert order.status == "unassigned" or order.status == "assigned"


def test_assign_orders_past_due():
    t = 50
    config = SimulationConfig()
    order = Order(t=0, r=10, d=40, loc=0)
    orders = [order]
    active_couriers = [(0, 0, 60)]

    SimulationUtils.assign_orders(t, orders, active_couriers, config)

    assert order.assigned_courier is None
    assert order.delivery_time is None
    assert order.status == "unassigned"
