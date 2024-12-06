import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from uber_network import uber_network
from map import city_map
from rider import uber_client

def run_and_plot_simulation(network):
    """
    Run the simulation using existing uber_network.run() and create plots
    
    Args:
        network (uber_network): Initialized uber_network instance
    """
    # Data collectors
    data = defaultdict(list)
    time_points = []
    last_collection_time = 0
    collection_interval = 1  # Collect data every simulation time unit
    
    # Store original event handlers
    original_handlers = {
        'client_arrival': network.handle_client_arrival,
        'driver_arrival': network.handle_driver_arrival,
        'pickup_start': network.handle_pickup_start,
        'pickup_end': network.handle_pickup_end,
        'ride_start': network.handle_ride_start,
        'ride_end': network.handle_ride_end,
        'driver_shift_end': network.handle_driver_shift_end
    }
    
    def collect_data():
        if network.current_time - last_collection_time >= collection_interval:
            time_points.append(network.current_time)
            queue_sizes = network.get_queue_sizes()
            data['main_queue'].append(queue_sizes['main_queue'])
            data['secondary_queue'].append(queue_sizes['secondary_queue'])
            
            # Get traffic density at center of map
            center_x, center_y = network.city_map.width // 2, network.city_map.height // 2
            traffic = network.city_map.get_traffic_density((center_x, center_y))
            data['traffic'].append(traffic)
            
            # Count active clients
            data['active_clients'].append(len(network.active_rides))
            
            # Calculate average wait and ride times
            if network.stats['total_matches'] > 0:
                avg_wait = network.stats['total_waiting_time'] / network.stats['total_matches']
                data['avg_wait'].append(avg_wait)
            else:
                data['avg_wait'].append(0)
                
            if network.stats['total_completed_rides'] > 0:
                avg_ride = network.stats['total_ride_time'] / network.stats['total_completed_rides']
                data['avg_ride'].append(avg_ride)
            else:
                data['avg_ride'].append(0)
            
            return True
        return False
    
    # Create wrapped handlers for all event types
    def wrap_handler(original_handler):
        def wrapped_handler(*args, **kwargs):
            nonlocal last_collection_time
            result = original_handler(*args, **kwargs)
            if collect_data():
                last_collection_time = network.current_time
            return result
        return wrapped_handler
    
    # Replace all handlers with wrapped versions
    network.handle_client_arrival = wrap_handler(original_handlers['client_arrival'])
    network.handle_driver_arrival = wrap_handler(original_handlers['driver_arrival'])
    network.handle_pickup_start = wrap_handler(original_handlers['pickup_start'])
    network.handle_pickup_end = wrap_handler(original_handlers['pickup_end'])
    network.handle_ride_start = wrap_handler(original_handlers['ride_start'])
    network.handle_ride_end = wrap_handler(original_handlers['ride_end'])
    network.handle_driver_shift_end = wrap_handler(original_handlers['driver_shift_end'])
    
    # Run simulation
    network.run()
    
    # Restore original handlers
    network.handle_client_arrival = original_handlers['client_arrival']
    network.handle_driver_arrival = original_handlers['driver_arrival']
    network.handle_pickup_start = original_handlers['pickup_start']
    network.handle_pickup_end = original_handlers['pickup_end']
    network.handle_ride_start = original_handlers['ride_start']
    network.handle_ride_end = original_handlers['ride_end']
    network.handle_driver_shift_end = original_handlers['driver_shift_end']
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Queue sizes over time
    ax1.plot(time_points, data['main_queue'], label='Main Queue')
    ax1.plot(time_points, data['secondary_queue'], label='Secondary Queue')
    ax1.set_title('Queue Sizes Over Time')
    ax1.set_xlabel('Simulation Time')
    ax1.set_ylabel('Queue Size')
    ax1.legend()
    
    # 2. Traffic density over time
    ax2.plot(time_points, data['traffic'])
    ax2.set_title('Traffic Density Over Time')
    ax2.set_xlabel('Simulation Time')
    ax2.set_ylabel('Traffic Density')
    
    # 3. Active clients over time
    ax3.plot(time_points, data['active_clients'])
    ax3.set_title('Active Clients Over Time')
    ax3.set_xlabel('Simulation Time')
    ax3.set_ylabel('Number of Active Clients')
    
    # 4. Average wait and ride times
    def moving_average(data, window=5):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Only plot non-zero values
    wait_times = np.array(data['avg_wait'])
    ride_times = np.array(data['avg_ride'])
    
    # Apply moving average if we have enough data points
    if len(wait_times) > 5:
        smoothed_wait = moving_average(wait_times)
        smoothed_ride = moving_average(ride_times)
        # Adjust time points for the moving average window
        plot_times = time_points[2:-2]
        ax4.plot(plot_times, smoothed_wait, label='Avg Wait Time')
        ax4.plot(plot_times, smoothed_ride, label='Avg Ride Time')
    else:
        ax4.plot(time_points, wait_times, label='Avg Wait Time')
        ax4.plot(time_points, ride_times, label='Avg Ride Time')
    
    ax4.set_title('Average Times Over Time')
    ax4.set_xlabel('Simulation Time')
    ax4.set_ylabel('Time Units')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    return data, time_points

# First create the map and network as before
my_map = city_map((10, 10), num_hotspots=3)
my_network = uber_network(
    city_map=my_map,
    client_arrival_rate=0.3,
    driver_arrival_rate=0.08,
    pre_simulation_driver=10,
    simulation_time=1000
)

# Pre-load some clients into queues before starting simulation
num_initial_clients = 100  # Adjust this number as needed

# Generate and add initial clients
for _ in range(num_initial_clients):
    # Create new client at time 0
    new_client = uber_client.generate_client(my_map, current_time=0)
    
    # Add to appropriate queue based on client type
    if new_client.behaviour_type == "Premium":
        my_network.main_queue.put(new_client)
    else:
        my_network.secondary_queue.put(new_client)
    
    # Update network stats
    my_network.stats['total_clients'] += 1

# Now run simulation and plot
data, time_points = run_and_plot_simulation(my_network)