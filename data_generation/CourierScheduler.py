import numpy as np
from typing import List, Tuple
from data_generation.simulation import SimulationConfig


class CourierScheduler:
    """Manages base and on-demand courier schedules with unique global IDs.

    Tracks all couriers in the simulation using persistent global IDs. Provides functionality
    for generating base schedules, adding on-demand couriers, and retrieving active couriers
    at any given time. Follows the paper’s experimental setup, supporting consistent courier
    identity across decision epochs for correct assignment and tracking.
    """

    def __init__(self, config: SimulationConfig):
        """Initializes the CourierScheduler with simulation parameters.

        Args:
            config: SimulationConfig object with courier settings and timing constants.
        """
        self.config = config
        self.next_courier_id = 0  # Global ID counter
        self.base_schedule = self._generate_base_schedule()
        self.on_demand_schedule = []  # On-demand couriers (with IDs)
        self.full_schedule = self.base_schedule[:]  # All couriers (base + on-demand)

    def _generate_base_schedule(self) -> List[Tuple[int, float, float]]:
        """Generates base couriers with persistent IDs and shift durations.

        Returns:
            List of tuples: (courier_id, start_time, end_time)
        """
        D1 = np.random.randint(self.config.D1_range[0], self.config.D1_range[1] + 1)
        D1_5 = np.random.randint(self.config.D1_5_range[0], self.config.D1_5_range[1] + 1)

        schedule = []

        # 1-hour couriers at specific base start times
        for t in [60, 120, 270, 330]:
            count = (D1 // 6) if t in [60, 120] else (D1 // 3)
            for _ in range(count):
                start = max(0, t + np.random.randint(-20, 21))
                end = start + 60
                schedule.append((self.next_courier_id, start, end))
                self.next_courier_id += 1

        # 1.5-hour couriers at different base start times
        for t in [0, 120, 240, 360]:
            count = D1_5 // 6
            for _ in range(count):
                start = max(0, t + np.random.randint(-20, 21))
                end = start + 90
                schedule.append((self.next_courier_id, start, end))
                self.next_courier_id += 1

        return schedule

    def add_on_demand_couriers(self, t: float, action: Tuple[int, int]) -> None:
        """Adds new on-demand couriers with global IDs.

        Args:
            t: Current time in minutes.
            action: Tuple (a1, a1_5), where a1 is the number of 1-hour couriers,
                    and a1_5 is the number of 1.5-hour couriers to add.
        """
        a1, a1_5 = action
        for c in [1] * a1 + [1.5] * a1_5:
            start = t + self.config.delta
            end = start + c * 60
            courier = (self.next_courier_id, start, end)
            self.on_demand_schedule.append(courier)
            self.full_schedule.append(courier)
            self.next_courier_id += 1

    def get_active_couriers_with_ids(self, t: float) -> List[Tuple[int, float, float]]:
        """Returns all active couriers with their global IDs at time t.

        Args:
            t: Current time in minutes.

        Returns:
            List of tuples (courier_id, start_time, end_time)
        """
        return [
            (cid, start, end)
            for cid, start, end in self.full_schedule
            if start <= t < end
        ]

    def get_active_couriers(self, t: float) -> int:
        """Returns the total number of active couriers at time t.

        Args:
            t: Current time in minutes.

        Returns:
            Integer count of active couriers.
        """
        return sum(1 for _, start, end in self.full_schedule if start <= t < end)

    def get_courier_changes(self, t: float, k: float) -> int:
        """Calculates net courier change in the window (t, t+k].

        Args:
            t: Current time in minutes.
            k: Length of the time window.

        Returns:
            Integer (starts - ends) for the given window.
        """
        starts = sum(1 for _, start, _ in self.full_schedule if t < start <= t + k)
        ends = sum(1 for _, _, end in self.full_schedule if t < end <= t + k)
        return starts - ends

    def reset(self) -> None:
        """Resets the scheduler state for a new simulation run."""
        self.next_courier_id = 0
        self.on_demand_schedule = []
        self.base_schedule = self._generate_base_schedule()
        self.full_schedule = self.base_schedule[:]
