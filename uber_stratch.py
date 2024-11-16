from dataclasses import dataclass
from enum import Enum
import random
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import queue

@dataclass
class Location:
    x:int
    y:int


# New Event System
class EventType(Enum):
    CLIENT_ARRIVAL = "CLIENT_ARRIVAL"
    DRIVER_ARRIVAL = "DRIVER_ARRIVAL"
    DRIVER_PICKUP_START = "DRIVER_PICKUP_START"
    DRIVER_PICKUP_END = "DRIVER_PICKUP_END"
    RIDE_START = "RIDE_START"
    RIDE_END = "RIDE_END"
    CLIENT_CANCELLATION = "CLIENT_CANCELLATION"
    DRIVER_SHIFT_END = "DRIVER_SHIFT_END"

@dataclass
class EventInfo:
    """Lightweight class to store essential event information"""
    client_id: Optional[str] = None
    driver_id: Optional[str] = None
    start_location: Optional[Tuple[int, int]] = None
    end_location: Optional[Tuple[int, int]] = None
    cancelled: bool = False

@dataclass
class Event:
    time: float
    event_type: EventType
    info: EventInfo
    
    def __lt__(self, other):
        return self.time < other.time

class FutureEventSet:
    def __init__(self):
        self.events = queue.PriorityQueue()
        self.current_time = 0.0
        
        # Reference dictionaries to store full objects
        self.client_registry: Dict[str, 'uber_client'] = {}
        self.driver_registry: Dict[str, 'uber_driver'] = {}
        
        # Keep track of pending events for cancellation
        self.pending_events: Dict[str, list[Event]] = {}
    
    def is_empty(self) -> bool:
        """Check if event queue is empty"""
        return self.events.empty()

    def register_client(self, client: 'uber_client') -> str:
        """Register a client and return its ID"""
        client_id = str(id(client))
        self.client_registry[client_id] = client
        self.pending_events[client_id] = []
        return client_id
    
    def register_driver(self, driver: 'uber_driver') -> str:
        """Register a driver and return its ID"""
        driver_id = str(id(driver))
        self.driver_registry[driver_id] = driver
        self.pending_events[driver_id] = []
        return driver_id
    
    def cleanup_registry(self, entity_id: str):
        """Clean up registry entries and pending events"""
        # Remove from registries
        self.client_registry.pop(entity_id, None)
        self.driver_registry.pop(entity_id, None)
        
        # Clean up pending events
        self.pending_events.pop(entity_id, None)
    
    def validate_event_time(self, time: float) -> float:
        """Validate and adjust event time if necessary"""
        if time < self.current_time:
            return self.current_time
        return time

    def add_event(self, event: Event):
        """Add a new event to the priority queue and track it"""
        # Validate time
        event.time = self.validate_event_time(event.time)
        
        # Add to queue
        self.events.put(event)
        
        # Track event for potential cancellation
        if event.info.client_id:
            self.pending_events[event.info.client_id].append(event)
        if event.info.driver_id:
            self.pending_events[event.info.driver_id].append(event)
    
    def cancel_entity_events(self, entity_id: str):
        """Cancel all pending events for a client or driver"""
        if entity_id in self.pending_events:
            for event in self.pending_events[entity_id]:
                event.info.cancelled = True
            self.pending_events[entity_id].clear()
    
    def get_next_event(self) -> Tuple[Optional[Event], Optional['uber_client'], Optional['uber_driver']]:
        """Get next valid event and associated objects"""
        while not self.events.empty():
            event = self.events.get()
            
            # Skip cancelled events
            if event.info.cancelled:
                continue
                
            self.current_time = event.time
            
            # Get associated objects
            client = self.get_client(event.info.client_id) if event.info.client_id else None
            driver = self.get_driver(event.info.driver_id) if event.info.driver_id else None
            
            return event, client, driver
            
        return None, None, None

    def get_client(self, client_id: str) -> Optional['uber_client']:
        """Retrieve client object from registry"""
        return self.client_registry.get(client_id)
    
    def get_driver(self, driver_id: str) -> Optional['uber_driver']:
        """Retrieve driver object from registry"""
        return self.driver_registry.get(driver_id)

    # Event scheduling methods
    def schedule_client_arrival(self, time: float, client: 'uber_client'):
        client_id = self.register_client(client)
        event = Event(
            time=time,
            event_type=EventType.CLIENT_ARRIVAL,
            info=EventInfo(
                client_id=client_id,
                start_location=(client.current_location.x, client.current_location.y)
            )
        )
        self.add_event(event)
    
    def schedule_driver_arrival(self, time: float, driver: 'uber_driver'):
        driver_id = self.register_driver(driver)
        event = Event(
            time=time,
            event_type=EventType.DRIVER_ARRIVAL,
            info=EventInfo(
                driver_id=driver_id,
                start_location=(driver.current_location.x, driver.current_location.y)
            )
        )
        self.add_event(event)
    
    def schedule_pickup_start(self, time: float, client: 'uber_client', driver: 'uber_driver'):
        event = Event(
            time=time,
            event_type=EventType.DRIVER_PICKUP_START,
            info=EventInfo(
                client_id=str(id(client)),
                driver_id=str(id(driver)),
                start_location=(driver.current_location.x, driver.current_location.y),
                end_location=(client.current_location.x, client.current_location.y)
            )
        )
        self.add_event(event)

    def schedule_pickup_end(self, time: float, client: 'uber_client', driver: 'uber_driver'):
        event = Event(
            time=time,
            event_type=EventType.DRIVER_PICKUP_END,
            info=EventInfo(
                client_id=str(id(client)),
                driver_id=str(id(driver)),
                start_location=(driver.current_location.x, driver.current_location.y),
                end_location=(client.current_location.x, client.current_location.y)
            )
        )
        self.add_event(event)

    def schedule_ride_start(self, time: float, client: 'uber_client', driver: 'uber_driver'):
        event = Event(
            time=time,
            event_type=EventType.RIDE_START,
            info=EventInfo(
                client_id=str(id(client)),
                driver_id=str(id(driver)),
                start_location=(client.current_location.x, client.current_location.y),
                end_location=(client.destination.x, client.destination.y)
            )
        )
        self.add_event(event)

    def schedule_ride_end(self, time: float, client: 'uber_client', driver: 'uber_driver'):
        event = Event(
            time=time,
            event_type=EventType.RIDE_END,
            info=EventInfo(
                client_id=str(id(client)),
                driver_id=str(id(driver)),
                end_location=(client.destination.x, client.destination.y)
            )
        )
        self.add_event(event)

    def schedule_client_cancellation(self, time: float, client: 'uber_client'):
        event = Event(
            time=time,
            event_type=EventType.CLIENT_CANCELLATION,
            info=EventInfo(client_id=str(id(client)))
        )
        self.add_event(event)

    def schedule_driver_shift_end(self, time: float, driver: 'uber_driver'):
        event = Event(
            time=time,
            event_type=EventType.DRIVER_SHIFT_END,
            info=EventInfo(driver_id=str(id(driver)))
        )
        self.add_event(event)

class uber_client:
    '''
    A client in the uber simulation 
    '''
    def __init__(self,arrival_time, current_location:Location, destination:Location, 
                 behavour_type= "Normal",max_wait_time:float = 15.0
                 ):
        ''' 
        Initialize a uber client(rider).

        args:
        todo
        '''
        # basic attributes 
        self.arrival_time = arrival_time
        self.current_location = current_location
        self.destination = destination
        self.behavour_type = behavour_type  
        self.max_wait_time = max_wait_time

        #status tracking
        self.status = "Waiting" # Waiting, Matched, InRide, Completed, Cancelled
        self.assigned_driver = None
        self.pickup_time = None
        self.completion_time = None

        # statistics
        self.waiting_time = 0
        self.ride_time = 0
        self.total_cost = 0

    @classmethod
    def generate_client(cls,city_map,current_time, behavior_dist = None):
        '''
        Generate a new client

        args:
            city_map
            arrival_time
            behavior_dist
        '''

        if behavior_dist == None:
            behavior_dist = {
                "Normal":0.7,
                "Premium":0.2,
                "Patient":0.1
            }
        
        grid_size = (city_map.width, city_map.height)
        start_loc = Location(
            x = random.randint(0,grid_size[0]-1),
            y = random.randint(0,grid_size[1]-1)
        )
        # ensure they are not the same pos
        while True:
            des_loc = Location(
            x = random.randint(0,grid_size[0]-1),
            y = random.randint(0,grid_size[1]-1)
        )
            if (des_loc.x != start_loc.x) or (des_loc.y != start_loc.y):
                break
        behavior = random.choices(
                    list(behavior_dist.keys()),
                    weights=list(behavior_dist.values())
                )[0]
        
        return cls(
            arrival_time = current_time,
            current_location = start_loc,
            destination = des_loc,
            behavour_type = behavior
        )
    
    def update_status(self, new_status: str, current_time: float):
        '''
        Update client status and calculate relevant times
        
        Args:
            new_status: New status to set
            current_time: Current simulation time
        '''
        self.status = new_status
        
        if new_status == "Matched":
            self.waiting_time = current_time - self.arrival_time
        elif new_status == "InRide":
            self.pickup_time = current_time
        elif new_status == "Completed":
            self.completion_time = current_time
            if self.pickup_time:
                self.ride_time = current_time - self.pickup_time

    def is_willing_to_wait(self,current_time):
        '''
        Check if client is still willing to wait based on their max wait time.
        
        Args:
            current_time (float): Current simulation time
        
        Returns:
            bool: True if client is still willing to wait, False otherwise

        '''

        current_wait  = current_time - self.arrival_time

        if self.behavour_type == "Patient":
            return current_wait <= (self.max_wait_time *1.3)
        elif self.behavour_type == "Premium":
            return current_wait <= (self.max_wait_time *0.7)
        
        current_time <= self.max_wait_time

class uber_driver:
    def __init__(self,current_location, ride_type= "UberX", status = "Idle"):
        '''
        Initialize a uber driver

        args:
        todo
        '''
        # basic attributes
        self.current_location = current_location
        self.behavour_type = ride_type
        self.status = status
        self.current_client = None

        # calculated attributes
        self.service_time = 0
        self.waiting_time = 0
    
    @classmethod
    def generate_driver(cls,city_map, ride_type_dist = None):
        '''
        Generate a new driver

        args:
            city_map
            arrival_time
            behavior_dist
        '''

        if ride_type_dist == None:
            ride_type_dist = {
                "UberX":0.7,
                "UberGreen":0.2,
                "UberLux":0.1
            }
        
        grid_size = (city_map.width, city_map.height)
        start_loc = Location(
            x = random.randint(0,grid_size[0]-1),
            y = random.randint(0,grid_size[1]-1)
        )
        
        ride_type = random.choices(
                    list(ride_type_dist.keys()),
                    weights=list(ride_type_dist.values())
                )[0]
        
        return cls(
            current_location = start_loc,
            ride_type = ride_type,
            status = "Idle"
        )

class uber_network:
    def __init__(self, city_map, client_arrival_rate, driver_arrival_rate, 
                 simulation_time: int = 1000,
                 base_score_threshold: float = 0.2,
                 secondary_queue_threshold: float = 0.1,
                 pre_simulation_driver: int = 20
                 ,seed: Optional[int] = None):
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
        
        #init queus and tracking
        self.main_queue = queue.Queue()
        self.secondary_queue = queue.Queue()
        self.available_drivers = [] # List of idle drivers
        self.active_rides = {}  # Dictionary to track ongoing rides
        self.completed_rides = [] # List to track completed rides
        
        # Initialize pre-simulation drivers
        self.drivers = []
        for _ in range(pre_simulation_driver):
            new_driver = uber_driver.generate_driver(self.city_map)
            self.drivers.append(new_driver)
            self.available_drivers.append(new_driver)
        
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
            # Schedule their shift ends
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
        """Handle a new client arrival"""
        self.stats['total_clients'] += 1
        self._try_matching_client(client)
        self._schedule_next_client_arrival()
        
        # Schedule potential cancellation
        if client.status == "Waiting":
            self.FES.schedule_client_cancellation(
                self.current_time + client.max_wait_time,
                client
            )

        
        # Schedule potential cancellation
        cancellation_time = event.time + client.max_wait_time
        self.FES.schedule_client_cancellation(cancellation_time, client)
    
    def handle_driver_arrival(self, event: Event, driver: uber_driver):
        """Handle a new driver arrival"""
        self.available_drivers.append(driver)
        self._schedule_next_driver_arrival()
        
        # Schedule shift end
        shift_duration = random.uniform(6, 12) * 3600
        self.FES.schedule_driver_shift_end(
            self.current_time + shift_duration,
            driver
        )

        # Try matching with waiting clients
        self._try_matching_queues()
    
    def handle_pickup_start(self, event: Event, client: uber_client, driver: uber_driver):
        """Handle driver starting to pick up client"""
        distance = self.city_map.calculate_distance(
            (driver.current_location.x, driver.current_location.y),
            (client.current_location.x, client.current_location.y)
        )
        traffic_factor = self.city_map.get_traffic_density(
            (client.current_location.x, client.current_location.y)
        )
        pickup_duration = distance * (1 + traffic_factor)
        
        self.FES.schedule_pickup_end(
            self.current_time + pickup_duration,
            client,
            driver
        )

    def handle_pickup_end(self, event: Event, client: uber_client, driver: uber_driver):
        """Handle driver arriving at client location"""
        driver.current_location = client.current_location
        self.FES.schedule_ride_start(self.current_time, client, driver)

    
    def handle_ride_start(self, event: Event, client: uber_client, driver: uber_driver):
        """Handle start of the ride"""
        client.update_status("InRide", self.current_time)
        driver.status = "InRide"
        
        distance = self.city_map.calculate_distance(
            (client.current_location.x, client.current_location.y),
            (client.destination.x, client.destination.y)
        )
        traffic_factor = self.city_map.get_traffic_density(
            (client.current_location.x, client.current_location.y)
        )
        ride_duration = distance * (1 + traffic_factor)
        
        self.FES.schedule_ride_end(
            self.current_time + ride_duration,
            client,
            driver
        )
    
    def handle_ride_end(self, event: Event, client: uber_client, driver: uber_driver):
        """Handle end of the ride"""
        # Update locations
        client.current_location = client.destination
        driver.current_location = client.destination
        
        # Update statuses
        client.update_status("Completed", self.current_time)
        driver.status = "Idle"
        
        # Update statistics
        self.stats['total_completed_rides'] += 1
        self.stats['total_ride_time'] += (self.current_time - client.pickup_time)
        
        # Make driver available again
        self.available_drivers.append(driver)
        
        # Remove from active rides
        if client in self.active_rides:
            del self.active_rides[client]
        
        # Try matching the driver with waiting clients
        self._try_matching_queues()

    def handle_client_cancellation(self, event: Event, client: uber_client):
        """Handle client cancellation due to wait time"""
        if client.status == "Waiting":
            client.update_status("Cancelled", self.current_time)
            self.stats['total_cancellations'] += 1
            
            # Remove from queues if present
            self._remove_from_queues(client)

    def handle_driver_shift_end(self, event: Event, driver: uber_driver):
        """Handle driver ending their shift"""
        if driver.status == "Idle":
            if driver in self.available_drivers:
                self.available_drivers.remove(driver)
            driver.status = "Off Shift"
    
    def _calculate_match_score(self, client: uber_client, driver: uber_driver) -> float:
        """Calculate a matching score between a client and driver"""
        # Calculate distance between driver and client
        distance = self.city_map.calculate_distance(
            (driver.current_location.x, driver.current_location.y),
            (client.current_location.x, client.current_location.y)
        )
        
        # Get traffic density at pickup location
        traffic_density = self.city_map.get_traffic_density(
            (client.current_location.x, client.current_location.y)
        )
        
        # Modified scoring to be more lenient
        base_score = 1.0 / (distance + 1)  # This gives 1.0 for distance=0, 0.5 for distance=1, etc.
        
        # Adjust traffic impact
        traffic_multiplier = 1.0 - (traffic_density * 0.5)  # Less impact from traffic
        
        # Adjust score based on client/driver type matching
        type_multiplier = 1.0
        if client.behavour_type == "Premium" and driver.behavour_type == "UberLux":
            type_multiplier = 1.5
        elif client.behavour_type == "Patient":
            type_multiplier = 1.2
            
        final_score = base_score * traffic_multiplier * type_multiplier
        print(f"Match score: {final_score:.2f} for distance: {distance}")  # Debug print
        return final_score
    
    def _try_matching_client(self, client: uber_client):
        """Try to match a new client with available drivers"""
        if not self.available_drivers:
            self.secondary_queue.put(client)
            return
            
        best_score = -1
        best_driver = None
        
        for driver in self.available_drivers:
            score = self._calculate_match_score(client, driver)
            if score > best_score:
                best_score = score
                best_driver = driver
        
        if best_score >= self.base_score_threshold:
            self._make_match(client, best_driver)
        else:
            self.secondary_queue.put(client)
    
    def _try_matching_queues(self):
        """Try matching waiting clients with available drivers"""
        self._try_matching_queue(self.main_queue, self.base_score_threshold)
        if self.available_drivers:  # If drivers still available
            self._try_matching_queue(self.secondary_queue, self.secondary_queue_threshold)
    

    
    def _try_matching_queue(self, queue_obj: queue.Queue, threshold: float):
        """Try matching clients in a specific queue"""
        if queue_obj.empty() or not self.available_drivers:
            return
            
        # Create temporary queue for unmatched clients
        temp_queue = queue.Queue()
        matched_count = 0
        
        while not queue_obj.empty():
            client = queue_obj.get()
            
            # Skip if client no longer waiting
            if client.status != "Waiting":
                continue
                
            best_score = -1
            best_driver = None
            
            for driver in self.available_drivers:
                score = self._calculate_match_score(client, driver)
                if score > best_score and score >= threshold:
                    best_score = score
                    best_driver = driver
            
            if best_driver:
                self._make_match(client, best_driver)
                matched_count += 1
            else:
                temp_queue.put(client)
        
        # Return unmatched clients to queue
        while not temp_queue.empty():
            queue_obj.put(temp_queue.get())
    
    def _make_match(self, client: uber_client, driver: uber_driver):
        """Complete a match between client and driver"""
        # Update statuses
        client.update_status("Matched", self.current_time)
        client.assigned_driver = driver
        driver.status = "Assigned"
        driver.current_client = client
        
        # Update tracking
        self.available_drivers.remove(driver)
        self.active_rides[client] = driver
        self.stats['total_matches'] += 1
        
        # Schedule pickup
        self.FES.schedule_pickup_start(self.current_time, client, driver)
    
    def _remove_from_queues(self, client: uber_client):
        """Remove a client from both queues"""
        # Helper function to remove from a specific queue
        def remove_from_queue(q):
            temp_queue = queue.Queue()
            found = False
            while not q.empty():
                c = q.get()
                if c != client:
                    temp_queue.put(c)
                else:
                    found = True
            while not temp_queue.empty():
                q.put(temp_queue.get())
            return found
        
        # Try removing from both queues
        remove_from_queue(self.main_queue)
        remove_from_queue(self.secondary_queue)
    
    def get_queue_sizes(self):
        """Get current sizes of both queues"""
        return {
            'main_queue': self.main_queue.qsize(),
            'secondary_queue': self.secondary_queue.qsize()
        }
    
    def _is_ride_complete(self, client: uber_client, driver: uber_driver) -> bool:
        """Check if a ride is complete based on time and distance"""
        if client.status != "InRide":
            return False
            
        distance = self.city_map.calculate_distance(
            (client.current_location.x, client.current_location.y),
            (client.destination.x, client.destination.y)
        )
        
        avg_traffic = (
            self.city_map.get_traffic_density((client.current_location.x, client.current_location.y)) +
            self.city_map.get_traffic_density((client.destination.x, client.destination.y))
        ) / 2
        
        estimated_duration = distance * (1 + avg_traffic)
        return (self.current_time - client.pickup_time) >= estimated_duration
    
    def _complete_ride(self, client: uber_client, driver: uber_driver):
        """Complete a ride and update all relevant status"""
        # Update client status and location
        client.update_status("Completed", self.current_time)
        client.current_location = client.destination
        
        # Update driver status and location
        driver.status = "Idle"
        driver.current_location = client.destination
        driver.current_client = None
        
        # Add driver back to available pool
        self.available_drivers.append(driver)
        
        # Add to completed rides
        self.completed_rides.append((client, driver))
    
    def run(self):
        """Run the simulation"""
        while not self.FES.is_empty() and self.FES.current_time < self.simulation_time:
            event, client, driver = self.FES.get_next_event()
            
            # Update current time
            self.current_time = event.time
            
            # Handle event based on type
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
            elif event.event_type == EventType.CLIENT_CANCELLATION:
                self.handle_client_cancellation(event, client)
            elif event.event_type == EventType.DRIVER_SHIFT_END:
                self.handle_driver_shift_end(event, driver)

    def get_statistics(self):
        """Get simulation statistics"""
        stats = self.stats.copy()
        
        # Calculate averages
        if stats['total_completed_rides'] > 0:
            stats['average_ride_time'] = stats['total_ride_time'] / stats['total_completed_rides']
        
        if stats['total_matches'] > 0:
            stats['average_wait_time'] = stats['total_waiting_time'] / stats['total_matches']
        
        stats['completion_rate'] = (stats['total_completed_rides'] / stats['total_clients'] 
                                  if stats['total_clients'] > 0 else 0)
        
        stats['cancellation_rate'] = (stats['total_cancellations'] / stats['total_clients']
                                    if stats['total_clients'] > 0 else 0)
        
        return stats

    def get_current_state(self):
        """Get current state of the system"""
        return {
            'time': self.current_time,
            'available_drivers': len(self.available_drivers),
            'active_rides': len(self.active_rides),
            'main_queue_size': self.main_queue.qsize(),
            'secondary_queue_size': self.secondary_queue.qsize(),
            'completed_rides': self.stats['total_completed_rides'],
            'total_matches': self.stats['total_matches'],
            'total_cancellations': self.stats['total_cancellations']
        }


class city_map:
    def __init__(self, grid_size, num_hotspots=3):
        """
        Initialize a grid-based city map with traffic hotspots.
        
        Args:
            grid_size (tuple): A tuple of (width, height) representing the size of the grid
            num_hotspots (int): Number of high-traffic areas to generate
        """
        self.width, self.height = grid_size
        self.road_network = self._create_road_network()
        self.traffic_density = self._initialize_traffic(num_hotspots)
        
    def _create_road_network(self):
        """Creates a dictionary representing the road network"""
        network = {}
        for x in range(self.width):
            for y in range(self.height):
                neighbors = []
                if x > 0:
                    neighbors.append((x-1, y))
                if x < self.width - 1:
                    neighbors.append((x+1, y))
                if y > 0:
                    neighbors.append((x, y-1))
                if y < self.height - 1:
                    neighbors.append((x, y+1))
                network[(x, y)] = neighbors
        return network
    
    def _initialize_traffic(self, num_hotspots):
        """Initialize traffic density map with hotspots"""
        density = defaultdict(lambda: 0.1)
        
        hotspots = []
        min_distance = min(self.width, self.height) // (num_hotspots + 1)
        
        while len(hotspots) < num_hotspots:
            x = random.randint(0, self.width-1)
            y = random.randint(0, self.height-1)
            new_spot = (x, y)
            
            if all(abs(x - spot[0]) + abs(y - spot[1]) >= min_distance 
                   for spot in hotspots):
                hotspots.append(new_spot)
                
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        pos_x = x + dx
                        pos_y = y + dy
                        if 0 <= pos_x < self.width and 0 <= pos_y < self.height:
                            distance = abs(dx) + abs(dy)
                            if distance == 0:
                                density[(pos_x, pos_y)] = 0.9
                            elif distance == 1:
                                density[(pos_x, pos_y)] = 0.7
                            else:
                                density[(pos_x, pos_y)] = 0.4
        
        return dict(density)
    
    def get_traffic_density(self, location):
        """
        Get traffic density at a specific location
        
        Args:
            location (tuple): Position as (x, y)
            
        Returns:
            float: Traffic density value between 0 and 1
        """
        x, y = location
        return self.traffic_density.get((x, y), 0.1)
    
    def get_neighbors(self, location):
        """Get all valid neighboring locations"""
        return self.road_network.get(location, [])
    
    def calculate_distance(self, start, end):
        """Calculate Manhattan distance between two points"""
        return abs(end[0] - start[0]) + abs(end[1] - start[1])
    
    def is_valid_location(self, location):
        """Check if a location is within bounds"""
        x, y = location
        return 0 <= x < self.width and 0 <= y < self.height
    
    def get_all_traffic_data(self):
        """
        Get complete traffic density map
        
        Returns:
            dict: Dictionary mapping (x,y) coordinates to traffic density values
        """
        return self.traffic_density

def get_traffic_level(density):
    """Helper function to convert density value to traffic level description"""
    if density >= 0.9:
        return "Very High"
    elif density >= 0.7:
        return "High"
    elif density >= 0.4:
        return "Moderate"
    else:
        return "Low"

def visualize_traffic_map(city):
    """
    Visualize the traffic density map.
    
    Args:
        city: Instance of city_map class
    """
    traffic_matrix = np.zeros((city.width, city.height))
    traffic_data = city.get_all_traffic_data()
    
    for (x, y), density in traffic_data.items():
        traffic_matrix[x][y] = density
    
    plt.figure(figsize=(10, 8))
    plt.imshow(traffic_matrix.T, cmap='YlOrRd', interpolation='nearest')
    plt.colorbar(label='Traffic Density')
    
    plt.grid(True, which='major', color='black', linewidth=0.5)
    plt.xticks(range(city.width))
    plt.yticks(range(city.height))
    
    plt.title('Traffic Density Map')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()


city = city_map((10, 10), num_hotspots=3)
current_time = 0.0

# Generate a new client
new_client = uber_client.generate_client(city, current_time)

new_driver = uber_driver.generate_driver(city)

def test_uber_queues():
    # Create a city map and network with pre-simulation drivers
    city_map_test = city_map((10, 10), num_hotspots=3)
    network = uber_network(city_map_test, 0.1, 0.05, pre_simulation_driver=5)
    
    current_time = 0.0
    
    # Print initial state with pre-simulation drivers
    print("Initial state with pre-simulation drivers:")
    print(f"Number of available drivers: {len(network.available_drivers)}")
    print(f"Main queue size: {network.main_queue.qsize()}")
    print(f"Secondary queue size: {network.secondary_queue.qsize()}")
    
    # Generate some test clients
    test_clients = []
    for i in range(5):
        client = uber_client.generate_client(city_map_test, current_time)
        test_clients.append(client)
    
    # Add clients with existing drivers
    print("\nAdding clients with pre-simulation drivers:")
    for i, client in enumerate(test_clients):
        network.add_client(client)
        print(f"After adding client {i+1}:")
        print(f"Available drivers: {len(network.available_drivers)}")
        print(f"Main queue size: {network.main_queue.qsize()}")
        print(f"Secondary queue size: {network.secondary_queue.qsize()}")
    
    # Now add some additional drivers
    print("\nAdding additional drivers:")
    for i in range(3):
        driver = uber_driver.generate_driver(city_map_test)
        network.add_driver(driver)
        print(f"After adding driver {i+1}:")
        print(f"Available drivers: {len(network.available_drivers)}")
        print(f"Main queue size: {network.main_queue.qsize()}")
        print(f"Secondary queue size: {network.secondary_queue.qsize()}")
        
    return network

# Run the test
network = test_uber_queues()