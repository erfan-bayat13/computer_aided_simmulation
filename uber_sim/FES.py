# New Event System
from dataclasses import dataclass
from enum import Enum
import queue
from typing import Dict, Optional, Tuple
from rider import uber_client
from driver import uber_driver

class EventType(Enum):
    CLIENT_ARRIVAL = "CLIENT_ARRIVAL"
    DRIVER_ARRIVAL = "DRIVER_ARRIVAL"
    DRIVER_PICKUP_START = "DRIVER_PICKUP_START"
    DRIVER_PICKUP_END = "DRIVER_PICKUP_END"
    RIDE_START = "RIDE_START"
    RIDE_END = "RIDE_END"
    #CLIENT_CANCELLATION = "CLIENT_CANCELLATION"
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
        """Register a client using their assigned ID"""
        self.client_registry[client.client_id] = client
        self.pending_events[client.client_id] = []
        return client.client_id
    
    def register_driver(self, driver: 'uber_driver') -> str:
        """Register a driver using their assigned ID"""
        self.driver_registry[driver.driver_id] = driver
        self.pending_events[driver.driver_id] = []
        return driver.driver_id
    
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
        
        # Initialize pending_events lists if they don't exist
        if event.info.client_id and event.info.client_id not in self.pending_events:
            self.pending_events[event.info.client_id] = []
            
        if event.info.driver_id and event.info.driver_id not in self.pending_events:
            self.pending_events[event.info.driver_id] = []
        
        # Track event for entities involved
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
        """
        Get next event with enhanced validation and error handling.
        """
        while not self.events.empty():
            event = self.events.get()
            
            # Skip cancelled events
            if event.info.cancelled:
                continue
                
            self.current_time = event.time
            
            # Get associated objects with validation
            client = None
            driver = None
            
            if event.info.client_id:
                client = self.client_registry.get(event.info.client_id)
                if not client and event.event_type != EventType.CLIENT_CANCELLATION:
                    print(f"Warning: Client {event.info.client_id} not found in registry for event {event.event_type}")
                    continue
                    
            if event.info.driver_id:
                driver = self.driver_registry.get(event.info.driver_id)
                if not driver:
                    print(f"Warning: Driver {event.info.driver_id} not found in registry for event {event.event_type}")
                    continue
            
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
        """Schedule client arrival with registration check"""
        if client.client_id not in self.client_registry:
            self.register_client(client)
            
        event = Event(
            time=time,
            event_type=EventType.CLIENT_ARRIVAL,
            info=EventInfo(
                client_id=client.client_id,
                start_location=(client.current_location.x, client.current_location.y)
            )
        )
        self.add_event(event)
    
    def schedule_driver_arrival(self, time: float, driver: 'uber_driver'):
        """Schedule driver arrival with registration check"""
        if driver.driver_id not in self.driver_registry:
            self.register_driver(driver)
            
        event = Event(
            time=time,
            event_type=EventType.DRIVER_ARRIVAL,
            info=EventInfo(
                driver_id=driver.driver_id,
                start_location=(driver.current_location.x, driver.current_location.y)
            )
        )
        self.add_event(event)

    
    def schedule_pickup_start(self, time: float, client: 'uber_client', driver: 'uber_driver'):
        """Schedule pickup start with registration checks"""
        if client.client_id not in self.client_registry:
            self.register_client(client)
        if driver.driver_id not in self.driver_registry:
            self.register_driver(driver)
            
        event = Event(
            time=time,
            event_type=EventType.DRIVER_PICKUP_START,
            info=EventInfo(
                client_id=client.client_id,
                driver_id=driver.driver_id,
                start_location=(driver.current_location.x, driver.current_location.y),
                end_location=(client.current_location.x, client.current_location.y)
            )
        )
        self.add_event(event)


    def schedule_pickup_end(self, time: float, client: 'uber_client', driver: 'uber_driver'):
        """Schedule pickup end with proper client and driver IDs"""
        event = Event(
            time=time,
            event_type=EventType.DRIVER_PICKUP_END,
            info=EventInfo(
                client_id=client.client_id,  # Use the proper ID attribute
                driver_id=driver.driver_id,  # Use the proper ID attribute
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
                client_id=client.client_id,
                driver_id=driver.driver_id,
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
                client_id=client.client_id,
                driver_id=driver.driver_id,
                start_location=(client.current_location.x, client.current_location.y),
                end_location=(client.destination.x, client.destination.y)
            )
        )
        self.add_event(event)

    '''
    def schedule_client_cancellation(self, time: float, client: 'uber_client'):
        # Validate that client exists and is registered
        if not client:
            print(f"Warning: Attempting to schedule cancellation for non-existent client at time {time}")
            return
            
        if client.client_id not in self.client_registry:
            print(f"Warning: Attempting to schedule cancellation for unregistered client {client.client_id}")
            return

        event = Event(
            time=time,
            event_type=EventType.CLIENT_CANCELLATION,
            info=EventInfo(
                client_id=client.client_id,  
                start_location=(client.current_location.x, client.current_location.y)  # Add location for tracking
            )
        )
        self.add_event(event)
        
        # Log cancellation scheduling
        print(f"""
        Scheduled Cancellation:
        - Time: {time}
        - Client ID: {client.client_id}
        - Current Status: {client.status}
        - Current Location: ({client.current_location.x}, {client.current_location.y})
        """)
    '''

    def schedule_driver_shift_end(self, time: float, driver: 'uber_driver'):
        """Schedule driver shift end with proper registration check"""
        # Ensure driver is registered first
        if driver.driver_id not in self.driver_registry:
            self.register_driver(driver)
            
        event = Event(
            time=time,
            event_type=EventType.DRIVER_SHIFT_END,
            info=EventInfo(driver_id=driver.driver_id)  # Use driver.driver_id instead of str(id(driver))
        )
        self.add_event(event)

    
    def get_registry_state(self):
        """Get current state of registries for debugging"""
        return {
            'clients': list(self.client_registry.keys()),
            'drivers': list(self.driver_registry.keys()),
            'pending_events': {k: len(v) for k, v in self.pending_events.items()}
        }