from collections import defaultdict
from dataclasses import dataclass
import random

@dataclass
class Location:
    x:int
    y:int
    
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