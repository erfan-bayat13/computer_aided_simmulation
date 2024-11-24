from dataclasses import dataclass
from enum import Enum
import math
import random
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import queue
from driver import uber_driver
from rider import uber_client
from map import city_map, Location
from FES import Event, EventType, EventInfo, FutureEventSet
class uber_network:
    def __init__(self, city_map, client_arrival_rate, driver_arrival_rate, 
                 simulation_time: int = 1000,
                 base_score_threshold: float = 0.2,
                 secondary_queue_threshold: float = 0.1,
                 pre_simulation_driver: int = 20,
                 min_drivers: int = 10,
                 seed: Optional[int] = None):
        '''
        Initialize the uber network

        args:
        todo
        '''
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        #basic attributes
            # Initialize FES
        self.FES = FutureEventSet()
        self.city_map = city_map
        self.client_arrival_rate = client_arrival_rate
        self.driver_arrival_rate = driver_arrival_rate
        self.base_score_threshold = base_score_threshold
        self.secondary_queue_threshold = secondary_queue_threshold
        self.current_time = 0.0 
        self.simulation_time = simulation_time
        self.min_drivers = min_drivers
        
        #init queus and tracking
        self.main_queue = queue.Queue()
        self.secondary_queue = queue.Queue()
        self.available_drivers = [] # List of idle drivers
        self.active_rides = {}  # Dictionary to track ongoing rides
        self.completed_rides = [] # List to track completed rides
        
        
        # Schedule first client arrival
        self._schedule_next_client_arrival()

        # Statistics tracking
        self.stats = {
            'total_matches': 0,
            'total_cancellations': 0,
            'secondary_queue_matches': 0,
            'total_waiting_time': 0,
            'total_ride_time': 0,
            'total_clients': 0,
            'total_completed_rides': 0
        }

        # Initialize pre-simulation drivers
        self._initialize_drivers(pre_simulation_driver)
        
        # Schedule first client arrival
        self._schedule_next_client_arrival()
        
        # Schedule first driver arrival
        self._schedule_next_driver_arrival()
    
    def _initialize_drivers(self, num_drivers: int):
        """Initialize the simulation with a set number of drivers"""
        for _ in range(num_drivers):
            new_driver = uber_driver.generate_driver(self.city_map)
            self.available_drivers.append(new_driver)
            # Register driver with FES first
            self.FES.register_driver(new_driver)
            # Then schedule shift end
            shift_duration = random.uniform(6, 12) * 3600  # 6-12 hours in seconds
            self.FES.schedule_driver_shift_end(self.current_time + shift_duration, new_driver)

    def _schedule_next_client_arrival(self):
        """Schedule the next client arrival using exponential distribution"""
        next_arrival = self.FES.current_time + random.expovariate(self.client_arrival_rate)
        new_client = uber_client.generate_client(self.city_map, next_arrival)
        self.FES.schedule_client_arrival(next_arrival, new_client)
    
    def _schedule_next_driver_arrival(self):
        """Schedule the next driver arrival"""
        next_arrival = self.current_time + random.expovariate(self.driver_arrival_rate)
        if next_arrival < self.simulation_time:
            new_driver = uber_driver.generate_driver(self.city_map)
            self.FES.schedule_driver_arrival(next_arrival, new_driver)
    
    def handle_client_arrival(self, event: Event, client: uber_client):
        """
        Handle a new client arrival with improved queue handling
        """
        # Ensure client is registered with FES
        if client.client_id not in self.FES.client_registry:
            self.FES.register_client(client)
            print(f"Client {client.client_id} registered with FES")
        
        client.status = "Waiting"
        client.arrival_time = self.current_time

        # Update statistics
        self.stats['total_clients'] += 1

        # Schedule next client arrival
        next_arrival = self.FES.current_time + random.expovariate(self.client_arrival_rate)
        print(f"Scheduling next client arrival at time: {next_arrival:.2f}")
        self._schedule_next_client_arrival()


        # Queue client based on type
        if client.behaviour_type == "Premium":
            self.main_queue.put(client)
            print(f"Premium client {client.client_id} added to main queue")
        else:
            self.secondary_queue.put(client)
            print(f"Client {client.client_id} added to secondary queue")
        
        '''
        if client.status == "Waiting":
            # Calculate when client might cancel
            cancellation_time = self.current_time + client.max_wait_time
            
            # Adjust cancellation time based on client type
            if client.behaviour_type == "Patient":
                cancellation_time *= 1.3
            elif client.behaviour_type == "Premium":
                cancellation_time *= 0.7
                
            # Schedule the potential cancellation event
            self.FES.schedule_client_cancellation(
                cancellation_time,
                client
            )
            
            #print(f"Client {client.client_id} cancellation scheduled for {cancellation_time}")
        
        # Try matching queues if there are available drivers
        # This will respect queue priority (main queue first, then secondary)
        if self.available_drivers:
            self._try_matching_queues()
        '''
        # Try matching queues if there are available drivers
        if self.available_drivers:
            print(f"Attempting to match - Available drivers: {len(self.available_drivers)}")
            self._try_matching_queues()
        else:
            print("No available drivers for immediate matching")
        

    def handle_driver_arrival(self, event: Event, driver: uber_driver):
        """
        Handle a new driver arrival into the system.
        
        This method:
        1. Validates and registers the new driver
        2. Initializes driver attributes and status
        3. Schedules their shift end
        4. Attempts to match with waiting clients
        5. Updates system statistics
        
        Args:
            event (Event): The driver arrival event
            driver (uber_driver): The arriving driver
        """
        # Validation
        if not driver:
            print(f"Error: Invalid driver arrival event at time {self.current_time}")
            return
            
        # Ensure driver is registered with FES
        if driver.driver_id not in self.FES.driver_registry:
            self.FES.register_driver(driver)
        
        # Initialize driver status if needed
        if driver.status not in ["Idle", "Off Shift"]:
            driver.status = "Idle"
        
        # Set initial location if not already set
        if not hasattr(driver, 'current_location') or driver.current_location is None:
            driver.current_location = Location(
                x=random.randint(0, self.city_map.width-1),
                y=random.randint(0, self.city_map.height-1)
            )
        
        # Add to available drivers if not already present
        if driver not in self.available_drivers:
            self.available_drivers.append(driver)
            
            # Update statistics
            self.stats['total_drivers'] = self.stats.get('total_drivers', 0) + 1
            self.stats['active_drivers'] = len(self.available_drivers)
            
            # Log driver arrival
            print(f"""
            Driver Arrival:
            - Time: {self.current_time}
            - Driver ID: {driver.driver_id}
            - Type: {driver.behaviour_type}
            - Location: ({driver.current_location.x}, {driver.current_location.y})
            - Active Drivers: {len(self.available_drivers)}
            """)
        
        # Schedule shift end
        shift_duration = random.uniform(6, 12) * 3600  # 6-12 hours in simulation time
        
        # Adjust shift duration based on driver type
        if driver.behaviour_type == "UberLux":
            shift_duration *= 0.8  # Luxury drivers tend to have shorter shifts
        elif driver.behaviour_type == "UberGreen":
            shift_duration *= 1.1  # Green drivers might work longer shifts
        
        # Add some randomness based on time of day (assuming 24-hour cycle)
        time_of_day = self.current_time % (24 * 3600)  # Convert to 24-hour cycle
        if 6 * 3600 <= time_of_day <= 9 * 3600:  # Morning rush (6 AM - 9 AM)
            shift_duration *= 1.2
        elif 16 * 3600 <= time_of_day <= 19 * 3600:  # Evening rush (4 PM - 7 PM)
            shift_duration *= 1.1
        
        # Schedule the shift end
        shift_end_time = self.current_time + shift_duration
        self.FES.schedule_driver_shift_end(
            shift_end_time,
            driver
        )
        
        # Log shift schedule
        print(f"""
        Shift Scheduled:
        - Driver: {driver.driver_id}
        - Duration: {shift_duration/3600:.2f} hours
        - End Time: {shift_end_time}
        """)
        
        # Try matching with waiting clients
        sizes_before = self.get_queue_sizes()
        self._try_matching_queues()
        sizes_after = self.get_queue_sizes()

        
        # Check if any matches were made
        matches_made = (
            (sizes_before['main_queue'] - sizes_after['main_queue']) +
            (sizes_before['secondary_queue'] - sizes_after['secondary_queue'])
        )
        
        if matches_made > 0:
            print(f"Driver {driver.driver_id} matched with {matches_made} waiting client(s)")
        
        # Schedule next driver arrival if needed
        if len(self.available_drivers) < self.min_drivers:
            self._schedule_next_driver_arrival()
        elif random.random() < 0.7:  # 70% chance to schedule next driver anyway
            self._schedule_next_driver_arrival()
        
        # Track driver distribution
        if not hasattr(self.stats, 'driver_type_distribution'):
            self.stats['driver_type_distribution'] = defaultdict(int)
        self.stats['driver_type_distribution'][driver.behaviour_type] += 1
        
        # Update peak tracking
        if not hasattr(self.stats, 'peak_active_drivers'):
            self.stats['peak_active_drivers'] = 0
        self.stats['peak_active_drivers'] = max(
            self.stats['peak_active_drivers'],
            len(self.available_drivers)
        )
        
        # Return success status
        return True
    
    def handle_pickup_start(self, event: Event, client: uber_client, driver: uber_driver):
        """
        Handle driver starting to pick up client.
        
        This function:
        1. Updates driver and client status
        2. Calculates pickup duration based on distance and traffic
        3. Schedules the pickup end event
        
        Args:
            event (Event): The pickup start event
            client (uber_client): The client to be picked up
            driver (uber_driver): The driver doing the pickup
        """
        # Validate that both client and driver exist and are in correct states
        if not client or not driver:
            print(f"Error: Missing client or driver for pickup start event at time {self.current_time}")
            return
            
        if client.status != "Matched" or driver.status != "Assigned":
            print(f"Error: Invalid status for pickup. Client: {client.status}, Driver: {driver.status}")
            return

        # Update statuses
        client.update_status("AwaitingPickup", self.current_time)
        driver.status = "EnRouteToClient"
        
        # Get pickup locations
        start_pos = (driver.current_location.x, driver.current_location.y)
        end_pos = (client.current_location.x, client.current_location.y)
        
        # Calculate pickup duration components
        distance = self.city_map.calculate_distance(start_pos, end_pos)
        
        # Get average traffic density along the route
        start_traffic = self.city_map.get_traffic_density(start_pos)
        end_traffic = self.city_map.get_traffic_density(end_pos)
        avg_traffic_density = (start_traffic + end_traffic) / 2
        
        # Calculate total pickup duration (distance * traffic factor)
        # Traffic factor increases time: 1.0 means no impact, 2.0 means doubles time
        traffic_factor = 1.0 + avg_traffic_density
        pickup_duration = distance * traffic_factor
        
        # Add some randomness to simulate real-world variations (±20%)
        variation = 0.8 + random.random() * 0.4  # Random factor between 0.8 and 1.2
        pickup_duration *= variation
        
        # Schedule the pickup end event
        pickup_end_time = self.current_time + pickup_duration
        self.FES.schedule_pickup_end(
            pickup_end_time,
            client,
            driver
        )
        
        # Log the pickup details for debugging/monitoring
        print(f"""
        Pickup Start Details:
        - Time: {self.current_time}
        - Client ID: {client.client_id}
        - Driver ID: {driver.driver_id}
        - Distance: {distance}
        - Traffic Density: {avg_traffic_density:.2f}
        - Estimated Duration: {pickup_duration:.2f}
        - Expected End Time: {pickup_end_time:.2f}
        """)

    def handle_pickup_end(self, event: Event, client: uber_client, driver: uber_driver):
        """
        Handle driver arriving at client location and completing pickup.
        
        This function:
        1. Validates the pickup completion
        2. Updates locations and statuses
        3. Calculates initial ride parameters
        4. Schedules the ride start
        
        Args:
            event (Event): The pickup end event
            client (uber_client): The client being picked up
            driver (uber_driver): The driver completing pickup
        """
        # Validation checks
        if not client or not driver:
            print(f"Error: Missing client or driver for pickup end event at time {self.current_time}")
            return
            
        if client.status != "AwaitingPickup" or driver.status != "EnRouteToClient":
            print(f"""
            Error: Invalid status for pickup end. 
            Client: {client.status}, Driver: {driver.status}
            Time: {self.current_time}
            """)
            return

        # Verify the driver has reached the client's location
        if (driver.current_location.x != client.current_location.x or 
            driver.current_location.y != client.current_location.y):
            # Update driver's location to client's location
            driver.current_location = Location(
                x=client.current_location.x,
                y=client.current_location.y
            )
        
        # Update statuses
        client.update_status("PickedUp", self.current_time)
        driver.status = "PickedUpClient"
        
        # Calculate initial ride parameters
        start_pos = (client.current_location.x, client.current_location.y)
        
        # Get traffic conditions for ride start
        current_traffic = self.city_map.get_traffic_density(start_pos)
        
        # Add some preparation time (30 seconds to 2 minutes in simulation time)
        prep_time = random.uniform(0.5, 2.0)
        ride_start_time = self.current_time + prep_time
        
        # Schedule ride start
        self.FES.schedule_ride_start(
            ride_start_time,
            client,
            driver
        )
        
        # Log the pickup completion and ride preparation
        print(f"""
        Pickup End Details:
        - Time: {self.current_time}
        - Client ID: {client.client_id}
        - Driver ID: {driver.driver_id}
        - Current Traffic: {current_traffic:.2f}
        - Preparation Time: {prep_time:.2f}
        - Ride Start Time: {ride_start_time:.2f}
        - Destination: ({client.destination.x}, {client.destination.y})
        """)
        
        # Update statistics if tracking pickup times
        pickup_duration = self.current_time - event.time
        if hasattr(self.stats, 'total_pickup_time'):
            self.stats['total_pickup_time'] = self.stats.get('total_pickup_time', 0) + pickup_duration
            self.stats['total_pickups'] = self.stats.get('total_pickups', 0) + 1

    
    def handle_ride_start(self, event: Event, client: uber_client, driver: uber_driver):
        """
        Handle the start of the ride after pickup is complete.
        
        This function:
        1. Validates ride can begin
        2. Updates statuses and locations
        3. Calculates ride parameters (duration, cost)
        4. Schedules ride end
        
        Args:
            event (Event): The ride start event
            client (uber_client): The client starting the ride
            driver (uber_driver): The driver starting the ride
        """
        # Validation checks
        if not client or not driver:
            print(f"Error: Missing client or driver for ride start event at time {self.current_time}")
            return
            
        if client.status != "PickedUp" or driver.status != "PickedUpClient":
            print(f"""
            Error: Invalid status for ride start. 
            Client: {client.status}, Driver: {driver.status}
            Time: {self.current_time}
            """)
            return
        
        # Verify client and driver are at same location
        if (driver.current_location.x != client.current_location.x or 
            driver.current_location.y != client.current_location.y):
            print(f"Warning: Driver and client locations don't match at ride start")
            # Sync locations to client's position
            driver.current_location = Location(
                x=client.current_location.x,
                y=client.current_location.y
            )
        
        # Update statuses
        client.update_status("InRide", self.current_time)
        driver.status = "InRide"
        
        # Calculate ride parameters
        start_pos = (client.current_location.x, client.current_location.y)
        end_pos = (client.destination.x, client.destination.y)
        distance = self.city_map.calculate_distance(start_pos, end_pos)
        
        # Calculate traffic impact
        start_traffic = self.city_map.get_traffic_density(start_pos)
        end_traffic = self.city_map.get_traffic_density(end_pos)
        # Check a midpoint for better traffic estimation
        mid_x = (client.current_location.x + client.destination.x) // 2
        mid_y = (client.current_location.y + client.destination.y) // 2
        mid_traffic = self.city_map.get_traffic_density((mid_x, mid_y))
        
        # Average traffic along route (weighted more towards start and mid points)
        avg_traffic = (0.4 * start_traffic + 0.4 * mid_traffic + 0.2 * end_traffic)
        
        # Calculate ride duration with traffic impact
        base_speed = 1.0  # Base units per time unit
        if driver.behaviour_type == "UberX":
            base_speed = 1.0
        elif driver.behaviour_type == "UberGreen":
            base_speed = 0.9  # Slightly slower
        elif driver.behaviour_type == "UberLux":
            base_speed = 1.1  # Slightly faster
            
        # Traffic factor increases time: 1.0 means no impact, 2.0 means doubles time
        traffic_factor = 1.0 + avg_traffic
        
        # Calculate total ride duration
        ride_duration = (distance / base_speed) * traffic_factor
        
        # Add some randomness to simulate real-world variations (±15%)
        variation = 0.85 + random.random() * 0.3
        ride_duration *= variation
        
        # Schedule ride end
        ride_end_time = self.current_time + ride_duration
        self.FES.schedule_ride_end(
            ride_end_time,
            client,
            driver
        )
        
        # Calculate preliminary cost
        base_rate = {
            "UberX": 1.0,
            "UberGreen": 1.2,
            "UberLux": 1.5
        }.get(driver.behaviour_type, 1.0)
        
        estimated_cost = distance * base_rate * (1 + 0.5 * avg_traffic)
        client.total_cost = estimated_cost  # Store estimated cost
        
        # Log ride start details
        print(f"""
        Ride Start Details:
        - Time: {self.current_time}
        - Client ID: {client.client_id}
        - Driver ID: {driver.driver_id}
        - Distance: {distance}
        - Average Traffic: {avg_traffic:.2f}
        - Estimated Duration: {ride_duration:.2f}
        - Expected End Time: {ride_end_time:.2f}
        - Estimated Cost: {estimated_cost:.2f}
        - Route: {start_pos} -> {end_pos}
        - Driver Type: {driver.behaviour_type}
        """)
        
        # Update statistics
        if hasattr(self.stats, 'total_rides_started'):
            self.stats['total_rides_started'] = self.stats.get('total_rides_started', 0) + 1
            self.stats['total_estimated_revenue'] = self.stats.get('total_estimated_revenue', 0) + estimated_cost
    
    def handle_ride_end(self, event: Event, client: uber_client, driver: uber_driver):
        """
        Handle the completion of a ride.
        
        This function:
        1. Validates ride completion
        2. Updates final locations
        3. Calculates final ride cost
        4. Updates statistics
        5. Manages driver availability
        6. Cleans up ride records
        
        Args:
            event (Event): The ride end event
            client (uber_client): The client completing the ride
            driver (uber_driver): The driver completing the ride
        """
        # Validation checks
        if not client or not driver:
            print(f"Error: Missing client or driver for ride end event at time {self.current_time}")
            return
            
        if client.status != "InRide" or driver.status != "InRide":
            print(f"""
            Error: Invalid status for ride end. 
            Client: {client.status}, Driver: {driver.status}
            Time: {self.current_time}
            """)
            return

        # Update locations to destination
        driver.current_location = Location(
            x=client.destination.x,
            y=client.destination.y
        )
        client.current_location = Location(
            x=client.destination.x,
            y=client.destination.y
        )

        # Calculate final ride metrics
        ride_duration = self.current_time - client.pickup_time
        actual_distance = self.city_map.calculate_distance(
            (event.info.start_location[0], event.info.start_location[1]),
            (event.info.end_location[0], event.info.end_location[1])
        )
        
        # Calculate final cost with potential adjustments
        base_rate = {
            "UberX": 1.0,
            "UberGreen": 1.2,
            "UberLux": 1.5
        }.get(driver.behaviour_type, 1.0)
        
        # Additional rate factors based on client type
        client_factor = {
            "Normal": 1.0,
            "Premium": 0.9,  # Premium clients get a discount
            "Patient": 0.95  # Patient clients get a small discount
        }.get(client.behaviour_type, 1.0)
        
        # Calculate final cost including time factor
        time_based_factor = min(ride_duration / client.pickup_time, 2.0)  # Cap at 2x
        final_cost = (actual_distance * base_rate * client_factor * 
                    (1 + 0.2 * time_based_factor))  # Time impacts cost less than distance
        
        # Update client cost
        client.total_cost = final_cost
        
        # Update statuses
        client.update_status("Completed", self.current_time)
        driver.status = "Idle"
        driver.current_client = None
        client.assigned_driver = None
        
        # Update driver's service time
        driver.service_time += ride_duration
        
        # Make driver available again
        self.available_drivers.append(driver)
        
        # Remove from active rides tracking
        self.active_rides.pop(client, None)
        
        # Update statistics
        self.stats['total_completed_rides'] = self.stats.get('total_completed_rides', 0) + 1
        self.stats['total_ride_time'] = self.stats.get('total_ride_time', 0) + ride_duration
        self.stats['total_revenue'] = self.stats.get('total_revenue', 0) + final_cost
        
        # Calculate and store ride metrics
        ride_metrics = {
            'client_id': client.client_id,
            'driver_id': driver.driver_id,
            'start_time': client.pickup_time,
            'end_time': self.current_time,
            'duration': ride_duration,
            'distance': actual_distance,
            'cost': final_cost,
            'client_type': client.behaviour_type,
            'driver_type': driver.behaviour_type
        }
        
        # Store ride metrics for analysis
        if not hasattr(self, 'completed_ride_metrics'):
            self.completed_ride_metrics = []
        self.completed_ride_metrics.append(ride_metrics)
        
        # Log ride completion details
        print(f"""
        Ride End Details:
        - Time: {self.current_time}
        - Client ID: {client.client_id} ({client.behaviour_type})
        - Driver ID: {driver.driver_id} ({driver.behaviour_type})
        - Ride Duration: {ride_duration:.2f}
        - Distance: {actual_distance:.2f}
        - Final Cost: {final_cost:.2f}
        - Start Location: {event.info.start_location}
        - End Location: {event.info.end_location}
        """)
        
        # Try to match the now-available driver with waiting clients
        self._try_matching_queues()
    '''
    def handle_client_cancellation(self, event: Event, client: uber_client):
        """
        Handle client cancellation due to wait time or other factors.
        
        This function:
        1. Validates cancellation is possible
        2. Updates client status
        3. Removes from queues
        4. Updates statistics
        5. Cleans up any pending matches
        
        Args:
            event (Event): The cancellation event
            client (uber_client): The client cancelling
        """
        # Validation checks
        if not client:
            print(f"Error: Missing client for cancellation event at time {self.current_time}")
            return
        
        if client.client_id not in self.FES.client_registry:
            print(f"Error: Client {client.client_id} not registered with FES for cancellation")
            return
            
        # Only process cancellation if client is still waiting
        if client.status != "Waiting":
            print(f"Note: Cancellation ignored - client {client.client_id} status is {client.status}")
            return
            
        # Calculate wait time before cancellation
        wait_time = self.current_time - client.arrival_time
        
        # Update client status
        client.update_status("Cancelled", self.current_time)
        
        # Remove from queues
        removed_from_main = self._remove_from_queue(self.main_queue, client)
        removed_from_secondary = self._remove_from_queue(self.secondary_queue, client)
        
        # Update statistics
        self.stats['total_cancellations'] = self.stats.get('total_cancellations', 0) + 1
        self.stats['total_wait_time_cancelled'] = self.stats.get('total_wait_time_cancelled', 0) + wait_time
        
        # Track cancellation metrics
        cancellation_data = {
            'client_id': client.client_id,
            'wait_time': wait_time,
            'client_type': client.behaviour_type,
            'time_of_day': self.current_time % 24,  # Assuming 24-hour cycle
            'queue_location': 'main' if removed_from_main else 'secondary' if removed_from_secondary else 'none'
        }
        
        if not hasattr(self, 'cancellation_metrics'):
            self.cancellation_metrics = []
        self.cancellation_metrics.append(cancellation_data)
        
        # Log cancellation details
        print(f"""
        Client Cancellation Details:
        - Time: {self.current_time}
        - Client ID: {client.client_id}
        - Client Type: {client.behaviour_type}
        - Wait Time: {wait_time:.2f}
        - Queue: {"Main" if removed_from_main else "Secondary" if removed_from_secondary else "None"}
        """)
        
        # Clean up from FES registry if needed
        self.FES.cleanup_registry(client.client_id)
    '''
    

    def handle_driver_shift_end(self, event: Event, driver: uber_driver):
        """
        Handle driver ending their shift.
        
        This function:
        1. Validates shift end is possible
        2. Updates driver status
        3. Removes from available pool
        4. Updates statistics
        5. Schedules new driver if needed
        
        Args:
            event (Event): The shift end event
            driver (uber_driver): The driver ending their shift
        """
        # Validation checks
        if not driver:
            print(f"Error: Missing driver for shift end event at time {self.current_time}")
            return
            
        # Only process shift end if driver is idle
        if driver.status != "Idle":
            # If driver is in middle of ride or pickup, schedule shift end for later
            if driver.status in ["InRide", "EnRouteToClient", "PickedUpClient"]:
                # Reschedule shift end for after estimated ride completion
                delay = random.uniform(10, 20)  # Add some buffer time
                self.FES.schedule_driver_shift_end(
                    self.current_time + delay,
                    driver
                )
                print(f"Driver {driver.driver_id} shift end postponed - currently {driver.status}")
                return
        
        # Calculate shift metrics
        shift_duration = self.current_time - event.time  # Time since last shift start
        
        # Update driver status
        driver.status = "Off Shift"
        
        # Remove from available drivers if present
        if driver in self.available_drivers:
            self.available_drivers.remove(driver)
        
        # Update statistics
        self.stats['total_shifts_completed'] = self.stats.get('total_shifts_completed', 0) + 1
        self.stats['total_shift_time'] = self.stats.get('total_shift_time', 0) + shift_duration
        
        # Track shift metrics
        shift_data = {
            'driver_id': driver.driver_id,
            'shift_duration': shift_duration,
            'service_time': driver.service_time,
            'driver_type': driver.behaviour_type,
            'time_of_day': self.current_time % 24  # Assuming 24-hour cycle
        }
        
        if not hasattr(self, 'shift_metrics'):
            self.shift_metrics = []
        self.shift_metrics.append(shift_data)
        
        # Log shift end details
        print(f"""
        Driver Shift End Details:
        - Time: {self.current_time}
        - Driver ID: {driver.driver_id}
        - Driver Type: {driver.behaviour_type}
        - Shift Duration: {shift_duration:.2f}
        - Service Time: {driver.service_time:.2f}
        - Utilization Rate: {(driver.service_time/shift_duration*100):.2f}%
        """)
        
        # Clean up from FES registry
        self.FES.cleanup_registry(driver.driver_id)
        
        # Check if we need to schedule a new driver
        if len(self.available_drivers) < self.min_drivers:
            self._schedule_next_driver_arrival()

        # Try to redistribute waiting clients if needed
        if self.available_drivers:
            self._try_matching_queues()

    def _remove_from_queue(self, queue_obj: queue.Queue, client: uber_client) -> bool:
        """
        Helper method to remove a client from a queue.
        
        Args:
            queue_obj: The queue to remove from
            client: The client to remove
            
        Returns:
            bool: True if client was found and removed, False otherwise
        """
        # Create temporary queue
        temp_queue = queue.Queue()
        found = False
        
        # Check each item in queue
        while not queue_obj.empty():
            current_client = queue_obj.get()
            if current_client != client:
                temp_queue.put(current_client)
            else:
                found = True
        
        # Restore remaining items
        while not temp_queue.empty():
            queue_obj.put(temp_queue.get())
            
        return found
    
    def _calculate_match_score(self, client: uber_client, driver: uber_driver) -> float:
        """
        Calculate a matching score between a client and driver with more lenient criteria.
        
        Args:
            client: The client to match
            driver: The driver to match
            
        Returns:
            float: Match score between 0 and 1, with higher values indicating better matches
        """
        # Calculate distance between driver and client
        distance = self.city_map.calculate_distance(
            (driver.current_location.x, driver.current_location.y),
            (client.current_location.x, client.current_location.y)
        )
        
        # Get traffic density at pickup location
        traffic_density = self.city_map.get_traffic_density(
            (client.current_location.x, client.current_location.y)
        )
        
        # More lenient base score calculation
        # Now uses exponential decay for distance
        # This gives higher scores for medium distances
        base_score = math.exp(-distance / 10.0)  # Slower decay rate
        
        # Softer traffic impact (reduced from 0.5 to 0.3)
        traffic_multiplier = 1.0 - (traffic_density * 0.3)
        
        # More generous type matching bonuses
        type_multiplier = 1.0
        if client.behaviour_type == "Premium":
            if driver.behaviour_type == "UberLux":
                type_multiplier = 1.5  # Increased from original
            elif driver.behaviour_type == "UberX":
                type_multiplier = 1.2  # Added bonus even for regular service
        elif client.behaviour_type == "Patient":
            type_multiplier = 1.3  # Increased from original
            
        # Calculate final score
        final_score = base_score * traffic_multiplier * type_multiplier
        
        # Enforce minimum score to make matches more likely
        final_score = max(final_score, 0.2)  # Guarantees minimum score of 0.2
        '''
        # Log scoring details for debugging
        print(f"""
        Match Score Calculation:
        - Client {client.client_id} ({client.behaviour_type}) to Driver {driver.driver_id} ({driver.behaviour_type})
        - Distance: {distance}
        - Base Score: {base_score:.3f}
        - Traffic Multiplier: {traffic_multiplier:.3f}
        - Type Multiplier: {type_multiplier:.3f}
        - Final Score: {final_score:.3f}
        """)
        '''
       
        
        return final_score
    
    def _try_matching_client(self, client: uber_client):
        """
        Try to match a new client with available drivers.
        
        This function:
        1. Checks available drivers
        2. Calculates match scores
        3. Picks best available match
        4. Either makes match or queues client
        
        Args:
            client (uber_client): The client to try matching
        
        Returns:
            bool: True if client was matched, False if queued
        """
        # First check if we have any available drivers
        if not self.available_drivers:
            # Put premium clients in main queue, others in secondary
            if client.behaviour_type == "Premium":
                self.main_queue.put(client)
                print(f"No drivers available. Premium client {client.client_id} added to main queue")
            else:
                self.secondary_queue.put(client)
                print(f"No drivers available. Client {client.client_id} added to secondary queue")
            return False

        # Initialize variables for best match
        best_score = -1
        best_driver = None
        match_scores = []  # Store all scores for logging

        # Calculate scores for all available drivers
        for driver in self.available_drivers:
            score = self._calculate_match_score(client, driver)
            match_scores.append((driver, score))
            if score > best_score:
                best_score = score
                best_driver = driver

        # Sort match scores for logging
        match_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Determine matching threshold based on client type and queue loads
        base_threshold = self.base_score_threshold
        
        # Adjust threshold based on client type
        if client.behaviour_type == "Premium":
            base_threshold *= 0.8  # 20% lower threshold for premium clients
        elif client.behaviour_type == "Patient":
            base_threshold *= 1.2  # 20% higher threshold for patient clients
        
        # Adjust threshold based on queue sizes
        total_waiting = self.main_queue.qsize() + self.secondary_queue.qsize()
        if total_waiting > 20:  # If many clients waiting
            base_threshold *= 0.9  # Lower threshold by 10%
        
        # Log matching attempt details
        print(f"""
        Matching Attempt for Client {client.client_id}:
        - Client Type: {client.behaviour_type}
        - Best Score: {best_score:.3f}
        - Threshold: {base_threshold:.3f}
        - Available Drivers: {len(self.available_drivers)}
        - Top 3 Scores: {[(driver.driver_id, f"{score:.3f}") for driver, score in match_scores[:3]]}
        """)

        # Make match if score is good enough
        if best_score >= base_threshold:
            self._make_match(client, best_driver)
            print(f"Successfully matched client {client.client_id} with driver {best_driver.driver_id} (score: {best_score:.3f})")
            return True
        else:
            # Queue client based on type and score
            if client.behaviour_type == "Premium" or best_score >= self.secondary_queue_threshold:
                self.main_queue.put(client)
                print(f"Client {client.client_id} added to main queue (score: {best_score:.3f})")
            else:
                self.secondary_queue.put(client)
                print(f"Client {client.client_id} added to secondary queue (score: {best_score:.3f})")
            
            # Store best potential match score for future reference
            if not hasattr(self, 'unmatched_scores'):
                self.unmatched_scores = []
            self.unmatched_scores.append({
                'client_id': client.client_id,
                'client_type': client.behaviour_type,
                'best_score': best_score,
                'threshold': base_threshold,
                'queue': 'main' if client.behaviour_type == "Premium" or 
                        best_score >= self.secondary_queue_threshold else 'secondary',
                'time': self.current_time
            })
            
            return False
    
    def _try_matching_queues(self):
        """
        Try matching waiting clients from both queues with available drivers.
        """
        # Quick check for available drivers
        if not self.available_drivers:
            print("No available drivers for matching")
            return
                
        # Log initial state
        initial_state = {
            'main_queue_size': self.main_queue.qsize(),
            'secondary_queue_size': self.secondary_queue.qsize(),
            'available_drivers': len(self.available_drivers)
        }
        
        print(f"\nStarting Queue Matching Process:")
        print(f"Main Queue Size: {initial_state['main_queue_size']}")
        print(f"Secondary Queue Size: {initial_state['secondary_queue_size']}")
        print(f"Available Drivers: {initial_state['available_drivers']}")

        self._rebalance_queues()
        
        # Process main queue first
        matches_made = self._try_matching_queue(
            self.main_queue,
            self.base_score_threshold,
            is_main_queue=True
        )
        
        # Process secondary queue
        if self.available_drivers:
            additional_matches = self._try_matching_queue(
                self.secondary_queue,
                self.secondary_queue_threshold,
                is_main_queue=False
            )
            matches_made += additional_matches
        
        print(f"Total matches made: {matches_made}")
    
    def _rebalance_queues(self):
        """
        Move long-waiting clients from secondary to main queue
        """
        if self.secondary_queue.empty():
            return
            
        # Store clients temporarily
        temp_clients = []
        promoted_count = 0
        
        # Base threshold for promotion (5 minutes in simulation time)
        wait_threshold = 100.0  
        
        while not self.secondary_queue.empty():
            client = self.secondary_queue.get()
            
            # Skip if client is no longer waiting
            if client.status != "Waiting":
                continue
                
            wait_time = self.current_time - client.arrival_time
            
            # Promote to main queue if waiting too long
            if wait_time > wait_threshold:
                self.main_queue.put(client)
                promoted_count += 1
                print(f"Promoted long-waiting client {client.client_id} to main queue (wait time: {wait_time:.1f})")
            else:
                temp_clients.append(client)
        
        # Return non-promoted clients to secondary queue
        for client in temp_clients:
            self.secondary_queue.put(client)
        
        if promoted_count > 0:
            print(f"Promoted {promoted_count} clients to main queue")

    def _try_matching_queue(self, queue_obj: queue.Queue, threshold: float, is_main_queue: bool) -> int:
        """
        Try matching clients from a specific queue.
        """
        if queue_obj.empty() or not self.available_drivers:
            return 0
                
        matches_made = 0
        temp_queue = queue.Queue()
        initial_size = queue_obj.qsize()
        
        print(f"\nProcessing {'Main' if is_main_queue else 'Secondary'} Queue:")
        print(f"Initial queue size: {initial_size}")
        print(f"Available drivers: {len(self.available_drivers)}")
        
        while not queue_obj.empty() and self.available_drivers:
            client = queue_obj.get()
            
            # Skip if client is not in waiting status
            if client.status != "Waiting":
                print(f"Skipping client {client.client_id} - status is {client.status}")
                continue
            
            # Find best available driver
            best_score = -1
            best_driver = None
            
            for driver in self.available_drivers:
                if driver.status != "Idle":
                    continue
                score = self._calculate_match_score(client, driver)
                if score > best_score:
                    best_score = score
                    best_driver = driver
            
            # Determine matching threshold
            adjusted_threshold = threshold
            if is_main_queue and client.behaviour_type == "Premium":
                adjusted_threshold *= 0.8  # Lower threshold for premium clients
            
            if best_score >= adjusted_threshold:
                print(f"Match found - Client: {client.client_id}, Driver: {best_driver.driver_id}, Score: {best_score:.3f}")
                self._make_match(client, best_driver)
                matches_made += 1
            else:
                print(f"No suitable match for client {client.client_id} - returning to queue")
                temp_queue.put(client)
        
        # Return unmatched clients to queue
        while not temp_queue.empty():
            queue_obj.put(temp_queue.get())
        
        print(f"Queue processing complete - Matches made: {matches_made}")
        return matches_made

    
    def _make_match(self, client: uber_client, driver: uber_driver):
        """
        Complete a match between client and driver, with proper FES integration.
        
        Args:
            client (uber_client): Client to be matched
            driver (uber_driver): Driver to be matched
        """
        # First ensure both client and driver are registered with FES
        if client.client_id not in self.FES.client_registry:
            self.FES.register_client(client)
        if driver.driver_id not in self.FES.driver_registry:
            self.FES.register_driver(driver)

        wait_time = self.current_time - client.arrival_time
        self.stats['total_waiting_time'] = self.stats.get('total_waiting_time', 0) + wait_time
        self.stats['total_matches'] = self.stats.get('total_matches', 0) + 1

        # Track individual wait time for analysis
        if not hasattr(self.stats, 'wait_times'):
            self.stats['wait_times'] = []
        self.stats['wait_times'].append(wait_time)
            
        # Update statuses and assignments
        client.update_status("Matched", self.current_time)
        client.assigned_driver = driver
        driver.status = "Assigned"
        driver.current_client = client
        
        # Remove driver from available pool
        if driver in self.available_drivers:
            self.available_drivers.remove(driver)
        
        # Update active rides tracking
        self.active_rides[client] = driver
        
        
        # Remove client from queues if present
        removal_results = self._remove_from_queues(client)
        if removal_results['original_queue'] == 'secondary':
            self.stats['secondary_queue_matches'] = self.stats.get('secondary_queue_matches', 0) + 1
        
        # Calculate initial pickup parameters
        start_pos = (driver.current_location.x, driver.current_location.y)
        end_pos = (client.current_location.x, client.current_location.y)
        
        # Schedule pickup start event
        self.FES.schedule_pickup_start(
            time=self.current_time,
            client=client,
            driver=driver
        )
        
        # Log match details
        print(f"""
        Match Created:
        - Time: {self.current_time}
        - Client: {client.client_id} ({client.behaviour_type})
        - Driver: {driver.driver_id} ({driver.behaviour_type})
        - Pickup Route: {start_pos} -> {end_pos}
        """)
    
    def _remove_from_queues(self, client: uber_client) -> dict:
        """
        Remove a client from both main and secondary queues.
        
        Args:
            client (uber_client): The client to remove from queues
            
        Returns:
            dict: Dictionary containing removal results:
                - 'main_queue_removed': True if client was in main queue
                - 'secondary_queue_removed': True if client was in secondary queue
                - 'original_queue': Which queue the client was found in ('main', 'secondary', or None)
        """
        results = {
            'main_queue_removed': False,
            'secondary_queue_removed': False,
            'original_queue': None
        }
        
        # Helper function to safely remove from a queue
        def remove_from_queue(queue_obj: queue.Queue, queue_name: str) -> bool:
            """
            Helper function to remove client from a specific queue
            
            Args:
                queue_obj: The queue to process
                queue_name: Name of queue for logging ('main' or 'secondary')
                
            Returns:
                bool: True if client was found and removed
            """
            temp_queue = queue.Queue()
            found = False
            original_size = queue_obj.qsize()
            
            # Process each client in queue
            while not queue_obj.empty():
                current_client = queue_obj.get()
                
                # Skip if this is our target client
                if current_client == client:
                    found = True
                    results['original_queue'] = queue_name
                    print(f"Client {client.client_id} removed from {queue_name} queue")
                    continue
                    
                # Otherwise, keep the client in queue
                temp_queue.put(current_client)
            
            # Restore remaining clients to original queue
            while not temp_queue.empty():
                queue_obj.put(temp_queue.get())
            
            # Log queue size change
            new_size = queue_obj.qsize()
            if found:
                print(f"{queue_name} queue size changed: {original_size} -> {new_size}")
                
            return found
        
        # Try removing from main queue
        results['main_queue_removed'] = remove_from_queue(self.main_queue, 'main')
        
        # Try removing from secondary queue
        results['secondary_queue_removed'] = remove_from_queue(self.secondary_queue, 'secondary')
        
        # Log overall results
        if results['original_queue']:
            print(f"""
            Queue Removal Results:
            - Client ID: {client.client_id}
            - Original Queue: {results['original_queue']}
            - Main Queue: {'Removed' if results['main_queue_removed'] else 'Not Found'}
            - Secondary Queue: {'Removed' if results['secondary_queue_removed'] else 'Not Found'}
            """)
        else:
            print(f"Client {client.client_id} not found in any queue")
        
        # If client was in neither queue, they might have been matched or cancelled
        if not any([results['main_queue_removed'], results['secondary_queue_removed']]):
            if client.status not in ['Matched', 'Cancelled', 'Completed']:
                print(f"Warning: Client {client.client_id} not in any queue but status is {client.status}")
        
        return results
    
    def get_queue_sizes(self):
        """
        Get basic queue size information for quick checks.
        
        Returns:
            dict: Simple dictionary with queue sizes
        """
        return {
            'main_queue': self.main_queue.qsize(),
            'secondary_queue': self.secondary_queue.qsize()
        }
    
    # not used
    def get_queue_analytics(self):
        """
        Get detailed analytics about queue state and waiting clients.
        
        Returns:
            dict: Comprehensive queue analytics including client types,
                 wait times, and other metrics
        """
        analytics = {
            'main_queue': {
                'size': self.main_queue.qsize(),
                'client_types': {'Premium': 0, 'Normal': 0, 'Patient': 0},
                'avg_wait_time': 0,
                'longest_wait': 0,
                'longest_waiting_client': None
            },
            'secondary_queue': {
                'size': self.secondary_queue.qsize(),
                'client_types': {'Premium': 0, 'Normal': 0, 'Patient': 0},
                'avg_wait_time': 0,
                'longest_wait': 0,
                'longest_waiting_client': None
            }
        }
        
        # Helper function to analyze a queue
        def analyze_queue(queue_obj, queue_type):
            if queue_obj.empty():
                return
                
            temp_queue = queue.Queue()
            total_wait_time = 0
            clients_processed = 0
            
            while not queue_obj.empty():
                client = queue_obj.get()
                wait_time = self.current_time - client.arrival_time
                
                # Update statistics
                analytics[queue_type]['client_types'][client.behavour_type] += 1
                total_wait_time += wait_time
                clients_processed += 1
                
                # Check for longest wait
                if wait_time > analytics[queue_type]['longest_wait']:
                    analytics[queue_type]['longest_wait'] = wait_time
                    analytics[queue_type]['longest_waiting_client'] = {
                        'client_id': client.client_id,
                        'type': client.behavour_type,
                        'waiting_since': client.arrival_time
                    }
                
                temp_queue.put(client)
            
            # Restore queue
            while not temp_queue.empty():
                queue_obj.put(temp_queue.get())
            
            # Calculate average wait time
            if clients_processed > 0:
                analytics[queue_type]['avg_wait_time'] = total_wait_time / clients_processed
        
        # Analyze both queues
        analyze_queue(self.main_queue, 'main_queue')
        analyze_queue(self.secondary_queue, 'secondary_queue')
        
        return analytics


    def _is_ride_complete(self, client: uber_client, driver: uber_driver) -> bool:
        """
        Check if a ride should be considered complete based on multiple factors.
        
        Args:
            client (uber_client): The client in the ride
            driver (uber_driver): The driver conducting the ride
            
        Returns:
            bool: True if ride should be considered complete, False otherwise
            
        Note:
            Takes into account:
            - Current status of client and driver
            - Distance traveled vs expected distance
            - Time elapsed vs expected duration
            - Traffic conditions along the route
            - Driver type and behavior
        """
        # Basic validation
        if (not client or not driver or 
            client.status != "InRide" or 
            driver.status != "InRide"):
            return False
        
        # Get current positions and destinations
        start_pos = (driver.current_location.x, driver.current_location.y)
        destination = (client.destination.x, client.destination.y)
        
        # Calculate total distance for the ride
        total_distance = self.city_map.calculate_distance(start_pos, destination)
        
        # Get traffic conditions along the route
        start_traffic = self.city_map.get_traffic_density(start_pos)
        end_traffic = self.city_map.get_traffic_density(destination)
        mid_x = (driver.current_location.x + client.destination.x) // 2
        mid_y = (driver.current_location.y + client.destination.y) // 2
        mid_traffic = self.city_map.get_traffic_density((mid_x, mid_y))
        
        # Calculate weighted average traffic (emphasize current location)
        avg_traffic = (0.5 * start_traffic + 0.3 * mid_traffic + 0.2 * end_traffic)
        
        # Adjust base speed by driver type
        base_speed = {
            "UberX": 1.0,
            "UberGreen": 0.9,
            "UberLux": 1.1
        }.get(driver.behaviour_type, 1.0)
        
        # Calculate expected duration with all factors
        traffic_factor = 1.0 + avg_traffic
        expected_duration = (total_distance / base_speed) * traffic_factor
        
        # Add buffer for various conditions
        # Premium clients get slightly faster service
        if client.behaviour_type == "Premium":
            expected_duration *= 0.9
        
        # Patient clients are okay with slightly longer rides
        elif client.behaviour_type == "Patient":
            expected_duration *= 1.1
        
        # Check if enough time has elapsed
        elapsed_time = self.current_time - client.pickup_time
        
        # Ride is complete if we've exceeded the expected duration
        # Add a small buffer (10%) to account for variations
        return elapsed_time >= (expected_duration * 1.1)
    
    def _complete_ride(self, client: uber_client, driver: uber_driver):
        """
        Complete a ride and handle all associated updates and cleanup.
        
        This method handles:
        - Status updates for client and driver
        - Location updates
        - Final cost calculations
        - Statistics updates
        - Queue and tracking cleanup
        - Driver availability management
        
        Args:
            client (uber_client): The client completing the ride
            driver (uber_driver): The driver completing the ride
        """
        # Validate ride can be completed
        if not client or not driver:
            print(f"Error: Missing client or driver for ride completion at time {self.current_time}")
            return
            
        if client.status != "InRide" or driver.status != "InRide":
            print(f"""
            Error: Invalid status for ride completion. 
            Client: {client.status}, Driver: {driver.status}
            Time: {self.current_time}
            """)
            return
        
        # Calculate final ride metrics
        ride_duration = self.current_time - client.pickup_time
        actual_distance = self.city_map.calculate_distance(
            (driver.current_location.x, driver.current_location.y),
            (client.destination.x, client.destination.y)
        )
        
        # Calculate final cost
        base_rate = {
            "UberX": 1.0,
            "UberGreen": 1.2,
            "UberLux": 1.5
        }.get(driver.behaviour_type, 1.0)
        
        # Apply client type adjustments
        client_factor = {
            "Normal": 1.0,
            "Premium": 0.9,  # Premium clients get a discount
            "Patient": 0.95  # Patient clients get a small discount
        }.get(client.behaviour_type, 1.0)
        
        # Get traffic conditions for final cost adjustment
        final_traffic = self.city_map.get_traffic_density((client.destination.x, client.destination.y))
        traffic_factor = 1.0 + (0.5 * final_traffic)  # Traffic affects price but not as much as base rate
        
        # Calculate final cost
        final_cost = (actual_distance * base_rate * client_factor * traffic_factor)
        
        # Update client status and attributes
        client.update_status("Completed", self.current_time)
        client.current_location = Location(
            x=client.destination.x,
            y=client.destination.y
        )
        client.ride_time = ride_duration
        client.total_cost = final_cost
        client.completion_time = self.current_time
        client.assigned_driver = None
        
        # Update driver status and attributes
        driver.status = "Idle"
        driver.current_location = Location(
            x=client.destination.x,
            y=client.destination.y
        )
        driver.current_client = None
        driver.service_time += ride_duration
        
        # Add driver back to available pool
        if driver not in self.available_drivers:
            self.available_drivers.append(driver)
        
        # Remove from active rides tracking
        if client in self.active_rides:
            self.active_rides.pop(client)
        
        # Create ride completion record
        ride_record = {
            'client_id': client.client_id,
            'driver_id': driver.driver_id,
            'pickup_time': client.pickup_time,
            'completion_time': self.current_time,
            'ride_duration': ride_duration,
            'distance': actual_distance,
            'final_cost': final_cost,
            'client_type': client.behaviour_type,
            'driver_type': driver.behaviour_type,
            'traffic_condition': final_traffic
        }
        
        # Add to completed rides list
        self.completed_rides.append(ride_record)
        
        # Update statistics
        self.stats['total_completed_rides'] = self.stats.get('total_completed_rides', 0) + 1
        self.stats['total_ride_time'] = self.stats.get('total_ride_time', 0) + ride_duration
        self.stats['total_revenue'] = self.stats.get('total_revenue', 0) + final_cost
        
        # Track average metrics
        if not hasattr(self.stats, 'ride_durations'):
            self.stats['ride_durations'] = []
        self.stats['ride_durations'].append(ride_duration)
        
        if not hasattr(self.stats, 'ride_distances'):
            self.stats['ride_distances'] = []
        self.stats['ride_distances'].append(actual_distance)
        
        if not hasattr(self.stats, 'ride_costs'):
            self.stats['ride_costs'] = []
        self.stats['ride_costs'].append(final_cost)
        
        # Log completion details
        print(f"""
        Ride Completed:
        - Time: {self.current_time}
        - Client: {client.client_id} ({client.behaviour_type})
        - Driver: {driver.driver_id} ({driver.behaviour_type})
        - Duration: {ride_duration:.2f}
        - Distance: {actual_distance:.2f}
        - Cost: {final_cost:.2f}
        - Traffic Factor: {traffic_factor:.2f}
        """)
        
        # Try to match the now-available driver with waiting clients
        self._try_matching_queues()
    
    def _cleanup_completed_entities(self):
        """Clean up completed or cancelled entities from FES registry"""
        # Clean up completed clients
        for client_id, client in list(self.FES.client_registry.items()):
            if client.status in ["Completed", "Cancelled"]:
                self.FES.cleanup_registry(client_id)
        
        # Clean up off-shift drivers
        for driver_id, driver in list(self.FES.driver_registry.items()):
            if driver.status == "Off Shift":
                self.FES.cleanup_registry(driver_id)

        # Optional: Log cleanup results
        if hasattr(self, 'debug_mode') and self.debug_mode:
            print(f"""
            Registry State After Cleanup:
            Clients: {len(self.FES.client_registry)}
            Drivers: {len(self.FES.driver_registry)}
            Pending Events: {sum(len(events) for events in self.FES.pending_events.values())}
            """)
    
    def run(self):
        """Run the simulation"""
        print(f"Starting simulation for {self.simulation_time} time units")
        
        # Initialize metrics for monitoring
        events_processed = 0
        last_report_time = 0
        report_interval = 100  # Report every 100 time units

        traffic_update_interval = 50  # Update traffic every 10 time units
        last_traffic_update = 0
        
        while True:
            # Get next event
            event, client, driver = self.FES.get_next_event()
            
            # Break conditions:
            # 1. No more events
            if event is None:
                print("No more events in queue")
                break
                
            # 2. Reached simulation time limit
            if event.time >= self.simulation_time:
                print(f"Reached simulation time limit: {self.simulation_time}")
                break
            
            # Update current time
            self.current_time = event.time

            # Update traffic conditions periodically
            if self.current_time - last_traffic_update >= traffic_update_interval:
                print(f"\n--- Updating Traffic at time {self.current_time} ---")
                self.city_map.update_traffic(self.current_time)
                last_traffic_update = self.current_time
            
            # Progress report
            if self.current_time - last_report_time >= report_interval:
                self._print_simulation_status()
                last_report_time = self.current_time
            
            # Handle event based on type
            try:
                if event.event_type == EventType.CLIENT_ARRIVAL:
                    self.handle_client_arrival(event, client)
                elif event.event_type == EventType.DRIVER_ARRIVAL:
                    self.handle_driver_arrival(event, driver)
                elif event.event_type == EventType.DRIVER_PICKUP_START:
                    self.handle_pickup_start(event, client, driver)
                elif event.event_type == EventType.DRIVER_PICKUP_END:
                    self.handle_pickup_end(event, client, driver)
                elif event.event_type == EventType.RIDE_START:
                    self.handle_ride_start(event, client, driver)
                elif event.event_type == EventType.RIDE_END:
                    self.handle_ride_end(event, client, driver)
                elif event.event_type == EventType.DRIVER_SHIFT_END:
                    self.handle_driver_shift_end(event, driver)

                # clean up the event from the FES
                if client and client.client_id in self.FES.pending_events:
                    self.FES.pending_events[client.client_id].remove(event)
                if driver and driver.driver_id in self.FES.pending_events:
                    self.FES.pending_events[driver.driver_id].remove(event)
                
                events_processed += 1
                
            except Exception as e:
                print(f"Error processing event {event.event_type}: {str(e)}")
                # Even if event handling fails, try to clean up
                if client and client.client_id in self.FES.pending_events:
                    self.FES.pending_events[client.client_id].remove(event)
                if driver and driver.driver_id in self.FES.pending_events:
                    self.FES.pending_events[driver.driver_id].remove(event)
                continue
            # clean up completed events
            self._cleanup_completed_entities()
        
        # Print final statistics
        print(f"\nSimulation completed:")
        print(f"Total events processed: {events_processed}")
        print(f"Final time: {self.current_time}")
        self._print_final_statistics()

    def _print_simulation_status(self):
        """Print current simulation status"""
        sizes = self.get_queue_sizes()
        print(f"\nStatus at time {self.current_time:.2f}:")
        print(f"Active drivers: {len(self.available_drivers)}")
        print(f"Active rides: {len(self.active_rides)}")
        print(f"Main queue size: {sizes['main_queue']}")
        print(f"Secondary queue size: {sizes['secondary_queue']}")

        # Add traffic monitoring for center and one corner of map
        sample_points = [
            (0, 0),
            (self.city_map.width//2, self.city_map.height//2),
        ]
        print("\nTraffic Samples:")
        for x, y in sample_points:
            density = self.city_map.get_traffic_density((x, y))
            print(f"Location ({x},{y}): {density:.3f}")
            
    def _print_final_statistics(self):
        """Print final simulation statistics with detailed wait time information"""
        stats = self.get_statistics()
        print("\nFinal Statistics:")
        print(f"Total clients: {stats['total_clients']}")
        print(f"Total matches: {stats['total_matches']}")
        print(f"Total completed rides: {stats['total_completed_rides']}")
        print(f"Total cancellations: {stats['total_cancellations']}")
        if stats['total_completed_rides'] > 0:
            print(f"Average ride time: {stats['average_ride_time']:.2f}")
        if stats['total_matches'] > 0:
            print(f"Average wait time: {stats['average_wait_time']:.2f}")
            if 'max_wait_time' in stats:
                print(f"Maximum wait time: {stats['max_wait_time']:.2f}")
                print(f"Minimum wait time: {stats['min_wait_time']:.2f}")
        print(f"Completion rate: {stats['completion_rate']*100:.2f}%")
        
    def get_statistics(self):
        """Get simulation statistics with improved wait time calculations"""
        stats = self.stats.copy()
        
        # Calculate averages
        if stats['total_completed_rides'] > 0:
            stats['average_ride_time'] = stats['total_ride_time'] / stats['total_completed_rides']
        
        if stats['total_matches'] > 0 and 'total_waiting_time' in stats:
            stats['average_wait_time'] = stats['total_waiting_time'] / stats['total_matches']
            
            if hasattr(self.stats, 'wait_times'):
                stats['max_wait_time'] = max(self.stats['wait_times'])
                stats['min_wait_time'] = min(self.stats['wait_times'])
        
        stats['completion_rate'] = (stats['total_completed_rides'] / stats['total_clients'] 
                                if stats['total_clients'] > 0 else 0)
        
        stats['cancellation_rate'] = (stats['total_cancellations'] / stats['total_clients']
                                    if stats['total_clients'] > 0 else 0)
        
        return stats