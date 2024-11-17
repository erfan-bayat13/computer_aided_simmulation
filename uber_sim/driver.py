import random
from map import Location

class uber_driver:
    _next_id = 1
    def __init__(self,current_location, ride_type= "UberX", status = "Idle"):
        '''
        Initialize a uber driver

        args:
        todo
        '''
        # Assign unique ID and increment counter
        self.driver_id = f"D{uber_driver._next_id}"
        uber_driver._next_id += 1
        
        # basic attributes
        self.current_location = current_location
        self.behaviour_type = ride_type
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
            ride_type_dist
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
    
    @classmethod
    def reset_id_counter(cls):
        """Reset the ID counter - useful for testing"""
        cls._next_id = 1