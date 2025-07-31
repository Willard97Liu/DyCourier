import sys
import os
import csv
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from data_generation.config import SimulationConfig
from data_generation.utils import SimulationUtils
from data_generation.order_generator import Order


def test_get_lost_orders_no_orders():
    t = 100
    config = SimulationConfig()
    visible_orders = []

    lost = SimulationUtils.get_lost_orders(t, visible_orders, config)

    assert lost == 0, "No orders should result in 0 lost"


def test_get_lost_orders_all_assignable():
    t = 100
    config = SimulationConfig()
    # ready at 90, due at 160; delivery takes 28 mins from now (pickup at 104, delivery at 132)
    o1 = Order(t=80, r=90, d=160, loc=0)
    o2 = Order(t=95, r=105, d=180, loc=1)
    visible_orders = [o1, o2]

    lost = SimulationUtils.get_lost_orders(t, visible_orders, config)

    assert lost == 0, "All orders can still be assigned and delivered"


def test_get_lost_orders_some_unreachable():
    t = 140
    config = SimulationConfig()
    # Due time too soon for delivery to finish
    o1 = Order(t=100, r=110, d=145, loc=0)  # 110 ready, delivery ends at 140+4+20+4=168 > 145 → lost
    o2 = Order(t=110, r=120, d=170, loc=1)  # 120 ready, delivery ends at 148 → ok

    visible_orders = [o1, o2]

    lost = SimulationUtils.get_lost_orders(t, visible_orders, config)

    assert lost == 1, "One order cannot be delivered in time"


def test_get_lost_orders_final_time_step():
    config = SimulationConfig()
    t = config.H0  # t = 450 (final order placement time)

    o1 = Order(t=400, r=410, d=460, loc=0)
    o2 = Order(t=420, r=430, d=470, loc=1)
    o3 = Order(t=430, r=440, d=480, loc=2)
    visible_orders = [o1, o2, o3]

    lost = SimulationUtils.get_lost_orders(t, visible_orders, config)

    assert lost == 3, "All unassigned orders at final timestep should be marked lost"


def test_get_lost_orders_assigned_orders_ignored():
    t = 200
    config = SimulationConfig()

    o1 = Order(t=150, r=160, d=240, loc=0)
    o1.assigned_courier = 1
    o1.delivery_time = 190

    o2 = Order(t=180, r=190, d=220, loc=1)
    visible_orders = [o1, o2]

    lost = SimulationUtils.get_lost_orders(t, visible_orders, config)

    assert lost == 1, "Only the unassigned order should be counted as potentially lost"
