import numpy as np
from typing import List, Tuple
from data_generation.simulation import SimulationConfig


# Order Generation
class OrderGenerator:
    """Generates orders based on the settings from the paper.

    This class creates orders for the rapid delivery system simulation, following the
    experimental settings from page 15 of the paper. It generates orders with placement
    times (t_o), ready times (e_o = t_o + 10), due times (m_o = t_o + 40), and pickup
    locations, with the number of orders per location drawn from a uniform distribution
    U[10,20]. Orders are uniformly distributed over the order placement period (H0=450 minutes).
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
        self.orders_per_location = np.random.randint(10, 21, config.N_pickup)
        # Total number of orders across all locations
        self.total_orders = sum(self.orders_per_location)
        # Generate order placement times uniformly over [0, H0]
        self.order_times = np.sort(np.random.uniform(0, config.H0, self.total_orders))
        # Ready times: e_o = t_o + 10 minutes (time when order is ready for pickup)
        self.order_ready_times = self.order_times + 10
        # Due times: m_o = t_o + 40 minutes (latest time for order delivery)
        self.order_due_times = self.order_times + 40
        # Assign each order to a pickup location
        self.order_locations = self._generate_locations()

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
        return list(
            zip(
                self.order_times,
                self.order_ready_times,
                self.order_due_times,
                self.order_locations,
            )
        )
