import random
from map import Location

class uber_client:
    '''
    A client in the uber simulation 
    '''
    _next_id = 1

    def __init__(self,arrival_time, current_location:Location, destination:Location, 
                 behaviour_type= "Normal",max_wait_time:float = 15.0
                 ):
        ''' 
        Initialize a uber client(rider).

        args:
        todo
        '''

        self.client_id = f"C{uber_client._next_id}"
        uber_client._next_id += 1
        # basic attributes 
        self.arrival_time = arrival_time
        self.current_location = current_location
        self.destination = destination
        self.behaviour_type = behaviour_type  
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
            behaviour_type = behavior
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
    '''
    def is_willing_to_wait(self,current_time):
        

        current_wait  = current_time - self.arrival_time

        if self.behaviour_type == "Patient":
            return current_wait <= (self.max_wait_time *1.3)
        elif self.behaviour_type == "Premium":
            return current_wait <= (self.max_wait_time *0.7)
        
        return current_wait <= self.max_wait_time
    '''
    
    
    @classmethod
    def reset_id_counter(cls):
        """Reset the ID counter - useful for testing"""
        cls._next_id = 1