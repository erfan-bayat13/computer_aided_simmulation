import random
import heapq

class SupermarketClient:
    """A client in a supermarket simulation."""
    
    def __init__(self, arrival_time, client_type="none", butchery_time=0, fresh_food_time=0, other_time=0,service_speed="normal"):
        """
        Initialize a supermarket client.
        
        Args:
            arrival_time (float): Time when client arrives at supermarket
            client_type (str): Type of client ("fresh", "butch", "both", "none")
            butchery_time (float): Service time needed at butchery
            fresh_food_time (float): Service time needed at fresh food section
            other_time (float): Time spent in other parts of supermarket
            service_speed (str): Speed of service ("normal" or "slow")
        """
        # Basic attributes
        self.arrival_time = arrival_time
        self.shopping_type = client_type
        self.service_speed = service_speed
        #self.item_bought = 0
        
        # Service time tracking
        base_service_times = {
            "butchery": butchery_time,
            "fresh_food": fresh_food_time,
            "other": other_time
        }

        cashier_time = self.calculate_cashier_time(base_service_times)

        # Complete service times including cashier
        self.service_times = {
            **base_service_times,
            "cashier": cashier_time
        }

        self.remaining_times = self.service_times.copy()
        
        # Location tracking
        self.current_location = "entrance"
        self.sections_visited = set()
        self.ready_for_checkout = False
        self.checked_out = False
        
        # Metrics tracking for each section
        self.section_metrics = {
            section: {
                "entry_time": None,
                "exit_time": None,
                "total_time": 0,
                "service_start": None
            }
            for section in ["butchery", "fresh_food", "other", "cashier"]
        }

            # Add patience attributes
        self.retry_attempts = {
            "butchery": 0,
            "fresh_food": 0,
            "other": 0,
            "cashier": 0
        }
        self.max_retries = 3  # Default patience threshold
        self.abandoned = False # Flag for abandoned clients
    
    @staticmethod
    def generate_random_client(arrival_time, mean_service_times=None):
        """
        Generate a random supermarket client.
        
        Args:
            arrival_time (float): Time when client arrives
            mean_service_times (dict): Mean service times for each section
        """
        if mean_service_times is None:
            mean_service_times = {
                "butchery": 5,
                "fresh_food": 4,
                "other": 15
            }
        # Determine if customer is slow (e.g., elderly, needs assistance)
        # 15% chance of being a slow customer
        service_speed = random.choices(["normal", "slow"], weights=[0.85, 0.15])[0]


        # Generate client type with weighted probabilities
        # Adjust probabilities for slow customers - they're more likely to need assistance
        if service_speed == "normal":
            client_type = random.choices(
                ["fresh", "butch", "both", "none"],
                weights=[0.3, 0.3, 0.2, 0.2]
            )[0]
        else:
            client_type = random.choices(
                ["fresh", "butch", "both", "none"],
                weights=[0.35, 0.35, 0.20, 0.10]
            )[0]
        
        # Generate service times based on client type
        butchery_time = random.expovariate(1/mean_service_times["butchery"]) if client_type in ["butch", "both"] else 0
        fresh_food_time = random.expovariate(1/mean_service_times["fresh_food"]) if client_type in ["fresh", "both"] else 0
        other_time = random.expovariate(1/mean_service_times["other"])
        
        return SupermarketClient(
            arrival_time=arrival_time,
            client_type=client_type,
            butchery_time=butchery_time,
            fresh_food_time=fresh_food_time,
            other_time=other_time,
            service_speed=service_speed
        )
    
    def calculate_cashier_time(self, service_times):
        """
        Calculate cashier service time based on other service times.
        
        The formula considers:
        1. Base time for scanning items (proportional to other shopping times)
        2. Additional time for specialized items (butchery/fresh food)
        3. Random variation to simulate different packing speeds, payment methods, etc.
        
        Args:
            service_times (dict): Dictionary of service times for other sections
            
        Returns:
            float: Calculated cashier service time
        """
        # Calculate base time proportional to total shopping time
        total_shopping_time = sum(service_times.values())
        base_time = total_shopping_time * 0.2  # Base scanning time is 20% of shopping time
        
        # Add extra time for specialized items
        specialized_time = 0
        if service_times["butchery"] > 0:
            specialized_time += service_times["butchery"] * 0.15  # 15% extra for butchery items
        if service_times["fresh_food"] > 0:
            specialized_time += service_times["fresh_food"] * 0.15  # 15% extra for fresh food items
            
        # Add random variation (±30% of base time)
        variation = random.uniform(0.7, 1.3)
        
        # Combine all components with a minimum time
        cashier_time = max(1.0, (base_time + specialized_time) * variation)
        
        return cashier_time
    
    def get_estimated_items(self):
        """
        Estimate the number of items based on service times.
        Useful for analysis and validation.
        
        Returns:
            dict: Estimated items per section
        """
        # Rough estimation of items based on service times
        items = {
            "butchery": int(self.service_times["butchery"] / 2) if self.service_times["butchery"] > 0 else 0,
            "fresh_food": int(self.service_times["fresh_food"] / 1.5) if self.service_times["fresh_food"] > 0 else 0,
            "other": int(self.service_times["other"] / 1) if self.service_times["other"] > 0 else 0
        }
        items["total"] = sum(items.values())
        return items
    
    @staticmethod
    def arrival(time, FES, waiting_queues, arrival_rate, metrics, supermarket):
        """
        Handle new client arrival.
        
        Args:
            time (float): Current time
            FES (list): Future Event Set
            waiting_queues (dict): Dictionary of queues for each section
            arrival_rate (float): Rate at which new clients arrive
            metrics (dict): Metrics tracking dictionary
            supermarket (SupermarketSimulator): Reference to main simulator
        """
        # Schedule next arrival
        inter_arrival_time = random.expovariate(arrival_rate)
        heapq.heappush(FES, (time + inter_arrival_time, "arrival", None))
        
        # Create new client
        new_client = SupermarketClient.generate_random_client(time)
        
        # Add client to supermarket tracking
        client_id = supermarket.client_counter + 1
        supermarket.client_counter = client_id
        supermarket.active_clients[client_id] = new_client
        
        # Determine initial section and schedule entry
        next_section = new_client.decide_next_section(waiting_queues)
        heapq.heappush(FES, (time, "section_entry", (client_id, next_section)))
        
        # Update metrics
        metrics['total_customers'] += 1
        return False  # Client is not rejected
    
    def decide_next_section(self, waiting_queues):
        """
        Decide which section to visit next.
        
        Args:
            waiting_queues (dict): Dictionary of current queue lengths
        
        Returns:
            str: Next section to visit or "exit" if done shopping
        """
        # If client has already checked out, they should exit
        if self.checked_out or self.current_location == "cashier":
            return "exit"
            
        # If ready for checkout, go to cashier
        if self.ready_for_checkout:
            return "cashier"
            
        # If in entrance, make initial decision
        if self.current_location == "entrance":
            # Check if specialized sections are too crowded
            queue_threshold = 5  # Maximum acceptable queue length
            
            if self.shopping_type in ["butch", "both"] and self.remaining_times["butchery"] > 0:
                if waiting_queues.get("butchery", 0) < queue_threshold:
                    return "butchery"
                else:
                    return "other"
            elif self.shopping_type in ["fresh", "both"] and self.remaining_times["fresh_food"] > 0:
                if waiting_queues.get("fresh_food", 0) < queue_threshold:
                    return "fresh_food"
                else:
                    return "other"
            else:
                return "other"
                
        # If in a specialized section (butchery or fresh food)
        elif self.current_location in ["butchery", "fresh_food"]:
            # Check if other specialized section is needed
            if (self.shopping_type == "both" and 
                self.current_location == "butchery" and 
                self.remaining_times["fresh_food"] > 0):
                return "fresh_food"
            elif (self.shopping_type == "both" and 
                  self.current_location == "fresh_food" and 
                  self.remaining_times["butchery"] > 0):
                return "butchery"
            elif self.remaining_times["other"] > 0:
                return "other"
            else:
                self.ready_for_checkout = True
                return "cashier"
        
        # If in other section
        elif self.current_location == "other":
            # If specialized sections still needed
            remaining_sections = []
            if self.remaining_times["butchery"] > 0 and self.shopping_type in ["butch", "both"]:
                remaining_sections.append("butchery")
            if self.remaining_times["fresh_food"] > 0 and self.shopping_type in ["fresh", "both"]:
                remaining_sections.append("fresh_food")
                
            if remaining_sections:
                # Choose section with shortest queue
                return min(remaining_sections, key=lambda x: waiting_queues.get(x, float('inf')))
            elif self.remaining_times["other"] > 0:
                return "other"
            else:
                self.ready_for_checkout = True
                return "cashier"
        
        return "exit"
    
    def get_service_time(self, section: str) -> float:
        """
        Get service time for a section with proper key handling.
        """
        return self.remaining_times[section] if section in self.remaining_times else 0
    
    def enter_section(self, time, section):
        """Record entry into a section."""
        self.current_location = section
        self.sections_visited.add(section)
        self.section_metrics[section]["entry_time"] = time
        self.section_metrics[section]["service_start"] = time

        # mark as checked out if client is in cashier
        if section == "cashier":
            self.checked_out = True
    
    def exit_section(self, time, section):
        """Record exit from a section."""
        metrics = self.section_metrics[section]
        metrics["exit_time"] = time
        
        if metrics["service_start"] is not None:
            service_time = time - metrics["service_start"]
            metrics["total_time"] += service_time
            
            # Update remaining time for section
            if section in self.remaining_times:
                self.remaining_times[section] = max(0, self.remaining_times[section] - service_time)
    
    def increment_retry_attempt(self, section):
        """Increment retry attempts for a specific section."""
        self.retry_attempts[section] = self.retry_attempts.get(section, 0) + 1
        return self.retry_attempts[section]

    def has_exceeded_patience(self, section):
        """Check if customer has exceeded patience threshold for a section."""
        return self.retry_attempts.get(section, 0) >= self.max_retries


    def get_total_shopping_time(self):
        """Calculate total time spent shopping."""
        return sum(metrics["total_time"] for metrics in self.section_metrics.values())
    
    def get_journey_summary(self):
        """Get summary of client's shopping journey."""
        return {
            'type': self.shopping_type,
            'sections_visited': list(self.sections_visited),
            'section_times': {
                section: metrics["total_time"]
                for section, metrics in self.section_metrics.items()
                if metrics["total_time"] > 0
            },
            'total_time': self.get_total_shopping_time()
        }