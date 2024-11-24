from collections import defaultdict
from dataclasses import dataclass
import math
import random
from typing import Dict, Tuple

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
        self.base_traffic_density = self.traffic_density.copy()  # Store initial state
        self.current_traffic_density = self.traffic_density.copy()
        self.hotspots = []
        self.events = []

        self.time_of_day = 0
        self.rush_hours = {
            'morning': (7, 9),
            'evening': (16, 18)
        }

        self.current_weather = 'clear'
        self.weather_factors = {
            'clear': 1.0,
            'rain': 1.3,
            'fog': 1.5
        }
        
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
    
    def update_traffic(self, current_time: float):
        """
        Update traffic conditions with more noticeable changes for shorter simulations.
        
        Args:
            current_time (float): Current simulation time
        """
        # Make a copy of the current traffic state
        new_traffic = self.traffic_density.copy()
        
        # Calculate time-based factor (oscillating pattern)
        # Using shorter time period for more noticeable changes
        time_factor = 1.0 + 0.3 * math.sin(current_time / 100.0)  # Shorter cycle
        
        # Apply time factor to all locations
        for pos in new_traffic:
            # Apply time-based changes
            new_traffic[pos] = new_traffic[pos] * time_factor
            
            # Add some random variation
            random_factor = 1.0 + random.uniform(-0.1, 0.1)
            new_traffic[pos] *= random_factor
            
            # Ensure values stay within bounds
            new_traffic[pos] = max(0.1, min(0.9, new_traffic[pos]))
        
        # Update the traffic state
        self.traffic_density = new_traffic
        
        # Print some debug info
        sample_pos = (self.width//2, self.height//2)  # Center of map
        print(f"\nTraffic Update at time {current_time:.1f}:")
        print(f"Time factor: {time_factor:.3f}")
        print(f"Center traffic: {self.traffic_density[sample_pos]:.3f}")
        
    def _get_time_factor(self) -> float:
        hour = self.time_of_day
        
        # Morning rush hour
        if self.rush_hours['morning'][0] <= hour < self.rush_hours['morning'][1]:
            # Gradual increase during morning rush
            progress = (hour - self.rush_hours['morning'][0]) / (self.rush_hours['morning'][1] - self.rush_hours['morning'][0])
            return 1.0 + (0.5 * math.sin(progress * math.pi))
        
        # Evening rush hour
        elif self.rush_hours['evening'][0] <= hour < self.rush_hours['evening'][1]:
            # Gradual increase during evening rush
            progress = (hour - self.rush_hours['evening'][0]) / (self.rush_hours['evening'][1] - self.rush_hours['evening'][0])
            return 1.0 + (0.6 * math.sin(progress * math.pi))
        
        # Night time reduction
        elif 22 <= hour or hour < 5:
            return 0.6
        
        # Normal daytime traffic
        return 1.0
    
    def _update_hotspots(self, current_time: float):
        """Update dynamic hotspots positions and intensities"""
        # Update existing hotspots
        for hotspot in self.hotspots:
            # Move hotspot slightly based on time
            dx = math.sin(current_time / 3600) * 0.1
            dy = math.cos(current_time / 3600) * 0.1
            
            hotspot['x'] = (hotspot['x'] + dx) % self.width
            hotspot['y'] = (hotspot['y'] + dy) % self.height
            
            # Vary intensity
            time_factor = 1 + 0.2 * math.sin(current_time / 7200)  # 2-hour cycle
            hotspot['intensity'] *= time_factor
            hotspot['intensity'] = max(0.2, min(0.8, hotspot['intensity']))
        
        # Occasionally add or remove hotspots
        if random.random() < 0.01:  # 1% chance each update
            if len(self.hotspots) < 5 and random.random() < 0.6:
                # Add new hotspot
                self.hotspots.append({
                    'x': random.randint(0, self.width-1),
                    'y': random.randint(0, self.height-1),
                    'intensity': random.uniform(0.3, 0.6),
                    'radius': random.randint(2, 4)
                })
            elif self.hotspots:
                # Remove random hotspot
                self.hotspots.pop(random.randint(0, len(self.hotspots)-1))
    
    def _calculate_hotspot_effects(self) -> Dict[Tuple[int, int], float]:
        """Calculate traffic effects from all hotspots"""
        effects = defaultdict(float)
        
        for hotspot in self.hotspots:
            center_x, center_y = int(hotspot['x']), int(hotspot['y'])
            radius = hotspot['radius']
            intensity = hotspot['intensity']
            
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    x, y = center_x + dx, center_y + dy
                    if 0 <= x < self.width and 0 <= y < self.height:
                        distance = math.sqrt(dx*dx + dy*dy)
                        if distance <= radius:
                            effect = intensity * (1 - distance/radius)
                            effects[(x, y)] += effect
        
        return effects
    
    def _update_events(self, current_time: float):
        """Update special events (accidents, construction, etc.)"""
        # Remove expired events
        self.events = [event for event in self.events 
                      if event[3] + event[4] > current_time]
        
        # Randomly add new events
        if random.random() < 0.005:  # 0.5% chance each update
            location = (random.randint(0, self.width-1), 
                       random.randint(0, self.height-1))
            radius = random.randint(1, 3)
            intensity = random.uniform(0.3, 0.7)
            duration = random.uniform(1800, 7200)  # 30 mins to 2 hours
            
            self.events.append((location, radius, intensity, 
                              current_time, duration))

    def _calculate_event_effects(self) -> Dict[Tuple[int, int], float]:
        """Calculate traffic effects from all active events"""
        effects = defaultdict(float)
        
        for location, radius, intensity, _, _ in self.events:
            center_x, center_y = location
            
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    x, y = center_x + dx, center_y + dy
                    if 0 <= x < self.width and 0 <= y < self.height:
                        distance = math.sqrt(dx*dx + dy*dy)
                        if distance <= radius:
                            effect = intensity * (1 - distance/radius)
                            effects[(x, y)] += effect
        
        return effects

    def _update_weather(self, current_time: float):
        """Update weather conditions periodically"""
        # Change weather every 4-8 hours (on average)
        if random.random() < 1/14400:  # Assuming time units are seconds
            weights = {
                'clear': 0.7,
                'rain': 0.2,
                'snow': 0.1
            }
            self.current_weather = random.choices(
                list(weights.keys()),
                weights=list(weights.values())
            )[0]
    
    def get_traffic_density(self, location):
        """
        Get traffic density at a specific location
        """
        x, y = location
        return self.current_traffic_density.get((x, y), 0.1)
    
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