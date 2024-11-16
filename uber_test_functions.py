from uber_stratch import *



def test_uber_queues():
    """Test the uber network simulation with focus on queue handling"""
    # Create a city map and network with pre-simulation drivers
    city_map_test = city_map((10, 10), num_hotspots=3)
    network = uber_network(city_map_test, 0.1, 0.05, pre_simulation_driver=5)
    
    # Print initial state with pre-simulation drivers
    print("Initial state:")
    print(f"Number of available drivers: {len(network.available_drivers)}")
    print(f"Main queue size: {network.main_queue.qsize()}")
    print(f"Secondary queue size: {network.secondary_queue.qsize()}")
    
    # Generate and schedule test clients
    print("\nScheduling client arrivals:")
    for i in range(5):
        client = uber_client.generate_client(city_map_test, network.current_time + i)
        network.FES.schedule_client_arrival(network.current_time + i, client)
        print(f"\nScheduled client {i+1} arrival at time {network.current_time + i}")
        
        # Process events until current time
        while not network.FES.is_empty() and network.FES.peek_next_time() <= network.current_time + i:
            event, client, driver = network.FES.get_next_event()
            network.current_time = event.time
            
            if event.event_type == EventType.CLIENT_ARRIVAL:
                network.handle_client_arrival(event, client)
                print(f"After handling client {i+1}:")
                print(f"Available drivers: {len(network.available_drivers)}")
                print(f"Main queue size: {network.main_queue.qsize()}")
                print(f"Secondary queue size: {network.secondary_queue.qsize()}")
    
    # Now add some additional drivers
    print("\nAdding additional drivers:")
    for i in range(3):
        driver = uber_driver.generate_driver(city_map_test)
        network.FES.schedule_driver_arrival(network.current_time + i, driver)
        
        # Process events until current time
        while not network.FES.events.empty() and network.FES.peek_next_time() <= network.current_time + i:
            event, client, driver = network.FES.get_next_event()
            network.current_time = event.time
            
            if event.event_type == EventType.DRIVER_ARRIVAL:
                network.handle_driver_arrival(event, driver)
                print(f"After adding driver {i+1}:")
                print(f"Available drivers: {len(network.available_drivers)}")
                print(f"Main queue size: {network.main_queue.qsize()}")
                print(f"Secondary queue size: {network.secondary_queue.qsize()}")
    
    return network

network = test_uber_queues()