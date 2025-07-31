from dataclasses import dataclass
from typing import List, Tuple, Dict
from data_generation.config import SimulationConfig


class SimulationUtils:
    """Utility functions for simulation tasks, including order assignment and lost order calculation.

    Supports the rapid delivery system simulation by providing methods to assign orders to couriers
    and compute lost orders, aligning with the paper 'Dynamic Courier Capacity Acquisition in Rapid
    Delivery Systems: A Deep Q-Learning Approach' (page 15). Includes realistic delivery durations
    with travel time (t_travel=20) to limit assignments, helping maintain non-zero q_orders during
    peak periods (180–300, 360–450 minutes).
    """
    #To avoid circular import 
    from data_generation.order_generator import Order

    @staticmethod
    def assign_orders(
        t: float,
        visible_orders: Order,
        id_active_couriers: List[Tuple[id, float, float]],  # (id, start_time, end_time)
        config: SimulationConfig,
    ):
        # 调试
        # print(f"现在时间：{t}")
        # for o in visible_orders:
        #     print(f"[订单] 下单时间: {o.order_time:.1f}, 截止时间: {o.due_time:.1f}, 分配骑手: {o.assigned_courier}, 送达时间: {o.delivery_time}")

        unassigned_orders = [o for o in visible_orders if o.assigned_courier is None]

        available_couriers = [
            (cid, start, end)
            for (cid, start, end) in id_active_couriers
            if start <= t < end
        ]

        ## 过滤掉正在派送订单的骑手，但是还要考虑送完订单的骑手，只要分配骑手，它的delivery_time就不为空
        assigned_courier_indices = {
            o.assigned_courier
            for o in visible_orders
            if o.assigned_courier is not None
            and o.delivery_time is not None
            and t < o.delivery_time
        }

        ## 用所有骑手减去这些正在派送订单的骑手
        available_couriers = [
            (i, start, end)
            for i, start, end in available_couriers
            if i not in assigned_courier_indices
        ]
        # 调试
        # print("[可用骑手]:")
        # for cid, start, end in available_couriers:
        #     if start <= t < end:
        #         print(f"ID={cid}, 在线时间=[{start}, {end})")

        # Step 3: 给目前未分配的订单分配骑手
        # 计算这些订单的最早开始可以去取的时间和计算这个订单，从最早现在去取，并且送完后的这个时间
        # 如果这个时间小于订单的due_time, 那么这个订单可以被分配，然后找这个订单是否超过车的due_time，如果没有超过
        # 则分配这个订单，给这个订单车的编号，并且记录这个订单的理想送达时间
        for order in sorted(unassigned_orders, key=lambda o: o.due_time):
            pickup_time = max(order.ready_time, t + config.s_p)
            delivery_end_time = pickup_time + config.t_travel + config.s_d

            if (
                delivery_end_time <= order.due_time
            ):  # 这个订单理论上能够分配，就看现在有没有合适的车了
                for idx, c_start, c_end in available_couriers:
                    if delivery_end_time <= c_end:  # 有合适的车
                        # 分配成功
                        order.assigned_courier = idx
                        order.delivery_time = delivery_end_time
                        # 在可以分配的骑手中移除该骑手
                        available_couriers = [
                            c for c in available_couriers if c[0] != idx
                        ]
                        break
        # 调试
        # for o in visible_orders:
        #     print(f"[订单] 下单时间: {o.order_time:.1f}, 截止时间: {o.due_time:.1f}, 分配骑手: {o.assigned_courier}, 送达时间: {o.delivery_time}")

    @staticmethod
    def get_lost_orders(
        t: float,
        visible_orders,
        config: SimulationConfig,
    ) -> int:
        """Calculates the number of orders lost between t and t_next."""

        lost = 0

        for o in visible_orders:
            if o.assigned_courier is not None:
                continue

            if t < config.H0:
                # 不是最后一刻，只判断当前出发也送不完的订单
                pickup_time = max(o.ready_time, t + config.s_p)
                delivery_end_time = pickup_time + config.t_travel + config.s_d
                if delivery_end_time > o.due_time:
                    lost += 1

            else:
                # 是最后一刻，所有还没被分配的订单都算 lost
                lost += 1

        return lost