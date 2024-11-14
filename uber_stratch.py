from dataclasses import dataclass
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class Location:
    x:int
    y:int

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
        self.assigened_client = None
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
                "nomral":0.7,
                "premium":0.2,
                "patient":0.1
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

class uber_driver:
    def __init__(self, current_location, behavour_type= "Normal", status = "Idle"):
        '''
        Initialize a uber driver

        args:
        todo
        '''
        # basic attributes
        self.current_location = current_location
        self.behavour_type = behavour_type
        self.status = status

        # calculated attributes
        self.service_time = 0
        self.waiting_time = 0

class uber_network:
    def __init__(self, city_map, traffic_generator):
        '''
        Initialize the uber network

        args:
        todo
        '''
        self.city_map = city_map
        self.traffic_generator = traffic_generator

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


city_map = city_map((10, 10), num_hotspots=3)
current_time = 0.0

# Generate a new client
new_client = uber_client.generate_client(city_map, current_time)
