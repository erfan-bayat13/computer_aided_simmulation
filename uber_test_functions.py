import random
import numpy as np
import matplotlib.pyplot as plt
from uber_stratch import *

def run_simulation_test(
    grid_size=(10, 10),
    simulation_time=10,
    client_arrival_rate=0.1,  # Average 1 client every 10 time units
    driver_arrival_rate=0.05,  # Average 1 driver every 20 time units
    num_hotspots=3,
    pre_simulation_drivers=20,
    seed=42
):
    """
    Run a complete test of the Uber simulation system.
    
    Args:
        grid_size: Tuple of (width, height) for the city grid
        simulation_time: Total time to run the simulation
        client_arrival_rate: Rate at which new clients appear
        driver_arrival_rate: Rate at which new drivers appear
        num_hotspots: Number of high-traffic areas
        pre_simulation_drivers: Number of drivers to start with
        seed: Random seed for reproducibility
    """
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    
    # Create city map and visualize initial traffic
    print("Initializing city map...")
    city = city_map(grid_size, num_hotspots)
    
    # Visualize initial traffic conditions
    plt.figure(figsize=(10, 8))
    traffic_data = np.zeros(grid_size)
    for (x, y), density in city.get_all_traffic_data().items():
        traffic_data[x][y] = density
    
    plt.imshow(traffic_data.T, cmap='YlOrRd', interpolation='nearest')
    plt.colorbar(label='Traffic Density')
    plt.title('Initial Traffic Density Map')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()
    
    # Initialize network
    print("\nInitializing Uber network...")
    network = uber_network(
        city_map=city,
        client_arrival_rate=client_arrival_rate,
        driver_arrival_rate=driver_arrival_rate,
        simulation_time=simulation_time,
        pre_simulation_driver=pre_simulation_drivers,
        seed=seed
    )
    
    # Run simulation
    print("\nStarting simulation...")
    network.run()
    
    # Get final statistics
    stats = network.get_statistics()
    
    # Print detailed statistics
    print("\nDetailed Statistics:")
    print("-" * 50)
    print(f"Total Simulation Time: {simulation_time}")
    print(f"Total Clients: {stats['total_clients']}")
    print(f"Total Matches: {stats['total_matches']}")
    print(f"Total Completed Rides: {stats['total_completed_rides']}")
    print(f"Total Cancellations: {stats['total_cancellations']}")
    
    if stats['total_completed_rides'] > 0:
        print(f"Average Ride Time: {stats['average_ride_time']:.2f}")
    
    if stats['total_matches'] > 0:
        print(f"Average Wait Time: {stats['average_wait_time']:.2f}")
    
    print(f"Completion Rate: {stats['completion_rate']*100:.2f}%")
    print(f"Cancellation Rate: {stats['cancellation_rate']*100:.2f}%")
    
    # Get final queue sizes
    queue_sizes = network.get_queue_sizes()
    print(f"\nFinal Queue States:")
    print(f"Main Queue Size: {queue_sizes['main_queue']}")
    print(f"Secondary Queue Size: {queue_sizes['secondary_queue']}")
    
    return network, stats

if __name__ == "__main__":
    # Run simulation with default parameters
    network, stats = run_simulation_test(
        grid_size=(10, 10),
        simulation_time=100,
        client_arrival_rate=0.1,
        driver_arrival_rate=0.05,
        num_hotspots=3,
        pre_simulation_drivers=20,
        seed=42
    )