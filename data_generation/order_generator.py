import numpy as np
from typing import List, Tuple
from data_generation.simulation import SimulationConfig


# Order Generation
class OrderGenerator:
    """Generates orders with peaks during lunch and dinner times based on the paper's settings.

    This class creates orders for the rapid delivery system simulation, following the
    experimental settings from page 15. It generates orders with placement times (t_o),
    ready times (e_o = t_o + 10), due times (m_o = t_o + 40), and pickup locations.
    The number of orders per location is drawn from U[10,20] for 16 locations.
    Order placement times are distributed over [0, H0=450] minutes with peaks during
    lunch (12:00 PM–2:00 PM) and dinner (3:00 PM–4:30 PM) to reflect realistic demand surges.
    """

    def __init__(self, config: SimulationConfig):
        """Initializes the OrderGenerator with simulation configuration.

        Args:
            config: SimulationConfig object containing parameters like N_pickup (number of pickup
                locations, default 16) and H0 (order placement period, default 450 minutes).
        """
        # Store configuration for access to parameters like N_pickup and H0
        self.config = config
        # Generate number of orders per pickup location, n_d ~ U[10,20], for N_pickup locations
        self.orders_per_location = np.random.randint(10, 20, config.N_pickup)
        # Total number of orders across all locations
        self.total_orders = sum(self.orders_per_location)
        # Generate order placement times with lunch and dinner peaks
        self.order_times = self._generate_order_times()
        # Ready times: e_o = t_o + 10 minutes (time when order is ready for pickup)
        self.order_ready_times = self.order_times + 10
        # Due times: m_o = t_o + 40 minutes (latest time for order delivery)
        self.order_due_times = self.order_times + 40
        # Assign each order to a pickup location
        self.order_locations = self._generate_locations()

    def _generate_order_times(self) -> np.ndarray:
        """Generates order placement times with peaks during lunch and dinner.

        Uses a mixture distribution: 50% of orders are uniformly distributed over [0, H0],
        25% follow a Gaussian distribution centered at 240 minutes (1:00 PM, lunch peak),
        and 25% follow a Gaussian centered at 405 minutes (3:45 PM, dinner peak). Times are
        clipped to [0, H0] and sorted to ensure chronological order.

        Returns:
            Sorted numpy array of order placement times (t_o) in minutes.
        """
        # Split total orders: 50% uniform, 25% lunch peak, 25% dinner peak
        n_uniform = self.total_orders // 2
        n_lunch = self.total_orders // 4
        n_dinner = self.total_orders - n_uniform - n_lunch  # Ensure total matches
        # Uniform distribution over [0, H0]
        uniform_times = np.random.uniform(0, self.config.H0, n_uniform)
        # Lunch peak: Gaussian centered at 240 minutes (1:00 PM, assuming 9:00 AM start)
        lunch_times = np.random.normal(loc=240, scale=30, size=n_lunch)
        # Dinner peak: Gaussian centered at 405 minutes (3:45 PM)
        dinner_times = np.random.normal(loc=405, scale=30, size=n_dinner)
        # Combine all times and clip to [0, H0]
        order_times = np.concatenate([uniform_times, lunch_times, dinner_times])
        order_times = np.clip(order_times, 0, self.config.H0)
        # Sort times to ensure chronological order
        return np.sort(order_times)

    def _generate_locations(self) -> List[int]:
        """Assigns pickup locations to orders randomly.

        Creates a list of location indices, where each location i has orders_per_location[i]
        orders. The locations are shuffled to randomize assignment to order times, ensuring
        a realistic distribution across the operating period.

        Returns:
            List of location indices (integers from 0 to N_pickup-1) for each order.
        """
        # Create a list with n_d orders for each location index i
        locations = []
        for i, n_d in enumerate(self.orders_per_location):
            locations.extend([i] * n_d)
        # Convert to numpy array for easier manipulation
        locations = np.array(locations)
        # Combine order times, ready times, due times, and locations into tuples
        order_data = list(
            zip(
                self.order_times,
                self.order_ready_times,
                self.order_due_times,
                locations,
            )
        )
        # Shuffle to randomize location assignment to order times
        np.random.shuffle(order_data)
        # Extract shuffled locations
        _, _, _, locations = zip(*order_data)
        return list(locations)

    def get_orders(self) -> List[Tuple[float, float, float, int]]:
        """Returns the list of generated orders.
        Returns:
            List of tuples, each containing (t_o, r_o, d_o, loc), where:
            - t_o: Order placement time (float, minutes).
            - r_o: Order ready time (t_o + 10, minutes).
            - d_o: Order due time (t_o + 40, minutes).
            - loc: Pickup location index (integer, 0 to N_pickup-1).
        """

        # Filter the arrays using the mask and return the zipped result
        return list(
            zip(
                self.order_times,
                self.order_ready_times,
                self.order_due_times,
                self.order_locations,
            )
        )
