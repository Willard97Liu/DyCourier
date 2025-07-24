import numpy as np
from data_generation.simulation import SimulationConfig
from typing import List, Tuple


# Courier Scheduling
class CourierScheduler:
    """Manages base and on-demand courier schedules for the rapid delivery system.

    This class handles the creation and maintenance of courier schedules, including
    pre-scheduled (base) couriers and dynamically added on-demand couriers. It follows
    the experimental settings from page 15 of the paper, with base courier counts drawn
    from uniform distributions (D1 ~ U[20,30] for 1-hour couriers, D1.5 ~ U[10,20] for
    1.5-hour couriers) and specific start times with random perturbations.
    """

    def __init__(self, config: SimulationConfig):
        """Initializes the CourierScheduler with simulation configuration.

        Args:
            config: SimulationConfig object containing parameters like D1_range, D1_5_range,
                and delta (show-up delay for on-demand couriers).
        """
        # Store configuration for access to parameters like courier ranges and delta
        self.config = config
        # Generate base courier schedule (pre-scheduled couriers)
        self.base_schedule = self._generate_base_schedule()
        # Initialize empty list for on-demand couriers added during simulation
        self.on_demand_schedule = []

    def _generate_base_schedule(self) -> List[Tuple[float, float]]:
        """Generates the base courier schedule according to the paper's settings.

        Creates a schedule for 1-hour and 1.5-hour couriers with counts drawn from
        D1 ~ U[20,30] and D1.5 ~ U[10,20]. Couriers are assigned start times at specific
        intervals (e.g., t=60, 120, 270, 330 for 1-hour couriers) with random perturbations
        in [-20, 20] minutes, ensuring start times are non-negative.

        Returns:
            List of tuples (start_time, courier_type), where start_time is in minutes and
            courier_type is 1 (1-hour) or 1.5 (1.5-hour).
        """

        D1 = np.random.randint(self.config.D1_range[0], self.config.D1_range[1] + 1)
        D1_5 = np.random.randint(
            self.config.D1_5_range[0], self.config.D1_5_range[1] + 1
        )

        schedule = []

        # 1-hour couriers at times t = 60, 120, 270, 330
        for t in [60, 120, 270, 330]:
            count = (D1 // 6) if t in [60, 120] else (D1 // 3)
            for _ in range(count):
                start = max(0, t + np.random.randint(-20, 21))
                schedule.append((start, start + 60))  # 1 hour

        # 1.5-hour couriers at times t = 0, 120, 240, 360
        for t in [0, 120, 240, 360]:
            count = D1_5 // 6
            for _ in range(count):
                start = max(0, t + np.random.randint(-20, 21))
                schedule.append((start, start + 90))  # 1.5 hours

        return schedule

    def add_on_demand_couriers(self, t: float, action: Tuple[int, int]) -> None:
        """Adds on-demand couriers to the schedule based on the action taken.

        On-demand couriers start at t + delta (e.g., 5 minutes) and work for their
        specified duration (1 or 1.5 hours). The action specifies how many couriers of
        each type to add.

        Args:
            t: Current time in minutes.
            action: Tuple (a1, a1_5), where a1 is the number of 1-hour couriers and
                a1_5 is the number of 1.5-hour couriers to add.
        """
        # Unpack action: number of 1-hour and 1.5-hour couriers
        a1, a1_5 = action
        # Create list of new couriers with start time t + delta and duration c (in minutes)
        new_couriers = [
            (t + self.config.delta, t + self.config.delta + c * 60)
            for c in [1] * a1 + [1.5] * a1_5
        ]
        # Append new couriers to on-demand schedule
        self.on_demand_schedule.extend(new_couriers)

    def get_active_couriers(self, t: float) -> int:
        """Counts the number of active couriers at time t.

        An active courier is one whose shift (start, end) includes time t, i.e., start <= t < end.
        Combines both base and on-demand couriers.

        Args:
            t: Current time in minutes.

        Returns:
            Integer number of active couriers, used as q_t^couriers in the state vector.
        """
        return sum(
            1
            for start, end in self.base_schedule + self.on_demand_schedule
            if start <= t < end
        )

    def get_courier_changes(self, t: float, k: float) -> int:
        """Calculates the net change in courier count in the time window (t, t+k].

        Computes the number of couriers starting minus those ending in the window (t, t+k],
        which is used to compute Theta_t^1 in the state vector, indicating future capacity changes.

        Args:
            t: Current time in minutes.
            k: Time window length in minutes (e.g., k1=120 for Theta_t^1).

        Returns:
            Integer net change (starts - ends) in courier count.
        """
        return sum(
            1
            for start, end in self.base_schedule + self.on_demand_schedule
            if t < start <= t + k
        ) - sum(
            1
            for start, end in self.base_schedule + self.on_demand_schedule
            if t < end <= t + k
        )
