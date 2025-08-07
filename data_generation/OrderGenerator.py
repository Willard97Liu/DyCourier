import numpy as np
from typing import List
from data_generation.Config import SimulationConfig


class Order:
    """
    Represents a delivery order in the simulation.

    Attributes:
        order_time (float): The time when the order was placed (t_o).
        ready_time (float): The time when the order is ready for pickup (e_o = t_o + 10).
        due_time (float): The latest time for delivery (m_o = t_o + 40).
        location (int): Index of the pickup location.
        assigned_courier (Optional[int]): ID of the assigned courier (None if unassigned).
        delivery_time (Optional[float]): Actual time of delivery (None if not delivered yet).
        status (str): Status of the order - "unassigned", "assigned", "delivered", or "lost".
    """

    def __init__(self, t, r, d, loc):
        self.order_time = t               # Order placement time
        self.ready_time = r              # Time when the order is ready for pickup
        self.due_time = d                # Latest delivery time
        self.location = loc              # Pickup location index
        self.assigned_courier = None     # Courier assigned to the order
        self.delivery_time = None        # Actual delivery time
        self.status = "unassigned"       # Initial order status

    def is_visible(self, t: float) -> bool:
        """
        Determines if the order is visible to the system at time t.

        Args:
            t (float): Current simulation time.

        Returns:
            bool: True if the order has been placed by time t.
        """
        return self.order_time <= t

    def is_active(self, t: float) -> bool:
        """
        Checks whether the order is active and awaiting delivery.

        Args:
            t (float): Current simulation time.

        Returns:
            bool: True if the order is ready and not yet delivered or overdue.
        """
        return self.ready_time <= t < self.due_time and self.status != "delivered"

    def is_lost(self, t: float) -> bool:
        """
        Determines whether the order has been lost (missed the due time).

        Args:
            t (float): Current simulation time.

        Returns:
            bool: True if the order is overdue and not delivered.
        """
        return t >= self.due_time and self.status != "delivered"


class OrderGenerator:
    """
    Generates delivery orders with peaks during lunch and dinner periods.

    This class creates a list of Order objects for use in a delivery simulation.
    Order placement times are distributed to reflect realistic customer demand patterns.
    """

    def __init__(self, config: SimulationConfig):
        """
        Initializes the order generator with the given simulation configuration.

        Args:
            config (SimulationConfig): Simulation parameters (e.g., number of pickup points and time horizon).
        """
        self.config = config

        # Randomly determine the number of orders for each pickup location (10–20 orders per location)
        self.orders_per_location = np.random.randint(10, 21, config.N_pickup)

        # Total number of orders to be generated
        self.total_orders = sum(self.orders_per_location)

        # Generate core components of the orders
        self.order_times = self._generate_order_times()
        self.order_ready_times = self.order_times + 10
        self.order_due_times = self.order_times + 40
        self.order_locations = self._generate_locations()

    def _generate_order_times(self) -> np.ndarray:
        """
        Generates order placement times with peaks at lunch and dinner.

        Returns:
            np.ndarray: Sorted array of order times clipped to the valid simulation period.
        """
        # Split orders into 50% uniform, 25% lunch peak, 25% dinner peak
        n_uniform = self.total_orders // 2
        n_lunch = self.total_orders // 4
        n_dinner = self.total_orders - n_uniform - n_lunch  # Remainder to ensure total matches

        # Uniformly distributed times over the full simulation horizon [0, H0]
        uniform_times = np.random.uniform(0, self.config.H0, n_uniform)

        # Lunch peak centered at 240 minutes (e.g., 1:00 PM assuming day starts at 9:00 AM)
        lunch_times = np.random.normal(loc=240, scale=30, size=n_lunch)

        # Dinner peak centered at 405 minutes (e.g., 3:45 PM)
        dinner_times = np.random.normal(loc=405, scale=30, size=n_dinner)

        # Combine all times and clip them to the valid range
        order_times = np.concatenate([uniform_times, lunch_times, dinner_times])
        order_times = np.clip(order_times, 0, self.config.H0)

        # Sort to maintain chronological order
        return np.sort(order_times)

    def _generate_locations(self) -> List[int]:
        """
        Randomly assigns pickup locations to each order based on pre-generated order counts.

        Returns:
            List[int]: Shuffled list of location indices corresponding to each order.
        """
        locations = []
        # Repeat each location index according to its assigned number of orders
        for i, n_d in enumerate(self.orders_per_location):
            locations.extend([i] * n_d)

        # Shuffle order data to mix location assignments across the timeline
        order_data = list(zip(self.order_times, self.order_ready_times, self.order_due_times, locations))
        np.random.shuffle(order_data)

        # Extract locations after shuffling
        _, _, _, shuffled_locations = zip(*order_data)
        return list(shuffled_locations)

    def get_orders(self) -> List[Order]:
        """
        Constructs the final list of Order objects using the generated attributes.

        Returns:
            List[Order]: List of Order instances ready for simulation.
        """
        return [
            Order(t, r, d, loc)
            for t, r, d, loc in zip(
                self.order_times,
                self.order_ready_times,
                self.order_due_times,
                self.order_locations,
            )
        ]
