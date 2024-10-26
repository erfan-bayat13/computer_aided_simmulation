import random
import heapq

class SupermarketClient:
    """A client in a supermarket simulation."""
    
    def __init__(self, arrival_time, client_type="none", butchery_time=0, fresh_food_time=0, other_time=0):
        """
        Initialize a supermarket client.
        
        Args:
            arrival_time (float): Time when client arrives at supermarket
            client_type (str): Type of client ("fresh", "butch", "both", "none")
            butchery_time (float): Service time needed at butchery
            fresh_food_time (float): Service time needed at fresh food section
            other_time (float): Time spent in other parts of supermarket
        """
        # Basic attributes
        self.arrival_time = arrival_time
        self.shopping_type = client_type
        
        # Service time tracking
        self.service_times = {
            "butchery": butchery_time,
            "fresh_food": fresh_food_time,
            "other": other_time
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
        
        # Generate client type with weighted probabilities
        client_type = random.choices(
            ["fresh", "butch", "both", "none"],
            weights=[0.3, 0.3, 0.2, 0.2]
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
            other_time=other_time
        )
    
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