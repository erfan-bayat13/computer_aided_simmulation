import heapq
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from queue import Queue
from typing import Dict, List, Optional
from client_supermarket import SupermarketClient

class SupermarketSimulator:
    def __init__(self, 
                 butchery_servers: int = 2,
                 fresh_food_servers: int = 2,
                 cashier_servers: int = 3,
                 queue_capacity: int = 10,
                 arrival_rate: float = 0.5,
                 simulation_time: float = 1000,
                 seed: Optional[int] = None):
        """Initialize the supermarket simulator with specified parameters."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Basic simulation parameters
        self.simulation_time = simulation_time
        self.arrival_rate = arrival_rate
        self.client_counter = 0
        self.active_clients = {}

        # Service rates for different sections
        self.service_rates = {
            "butchery": 0.4,     # Avg service time = 2.5 time units
            "fresh_food": 0.5,   # Avg service time = 2 time units
            "other": 0.3,        # Avg service time = 3.33 time units
            "cashier": 0.6       # Avg service time = 1.67 time units
        }

        # Initialize sections with special handling for "other"
        self.sections = {
            "butchery": {
                "servers": [None] * butchery_servers,
                "queue": Queue(maxsize=queue_capacity)
            },
            "fresh_food": {
                "servers": [None] * fresh_food_servers,
                "queue": Queue(maxsize=queue_capacity)
            },
            "cashier": {
                "servers": [None] * cashier_servers,
                "queue": Queue(maxsize=queue_capacity)
            }
        }
        
        # Special handling for "other" section - no queue needed as it has infinite servers
        self.sections["other"] = {
            "servers": [],  # Dynamic list that can grow as needed
            "queue": Queue(maxsize=0)  # Zero-capacity queue as it's not needed
        }

        # Initialize metrics 
        self.metrics = self.initialize_metrics()

        # Initialize FES with first arrival
        self.FES = []
        heapq.heappush(self.FES, (0, "arrival", None))
    
    def initialize_metrics(self):
        """Initialize the metrics dictionary with all required fields."""
        def create_hour_stats():
            return {
                'arrivals': 0,
                'completions': 0,
                'avg_time': 0.0,
                'total_time': 0.0
            }
        return {
            'total_customers': 0,          # Total customers who entered the system
            'active_customers': 0,         # Currently in the system
            'completed_customers': 0,      # Successfully exited after all services
            'rejected_customers': 0,       # Rejected at any point
            'total_time_in_system': [],    # Time from entry to exit
            'customer_paths': [],          # Sections visited by each customer
            'customer_journeys': [],       # Detailed journey info
            'hourly_statistics': defaultdict(create_hour_stats),
            'section_metrics': {
                section: {
                    'current_customers': 0,     # Currently in section
                    'total_arrivals': 0,        # Total arrivals to section
                    'completions': 0,           # Completed service in section
                    'service_times': [],        # Time spent in service
                    'waiting_times': [],        # Time spent waiting
                    'queue_lengths': [],        # Queue length measurements
                    'server_utilization': [],   # Added this field
                    'rejected_customers': 0,    # Rejected from section
                    'total_times': [],           # Total time in section (waiting + service)
                    'unique_customers': set()   # Unique customers served
                } for section in self.sections
            }
        }
    
    def update_hourly_statistics(self, time: float, event_type: str, total_time: float = None):
        """Update hourly statistics with proper initialization."""
        hour = int(time // 60)
        stats = self.metrics['hourly_statistics'][hour]
        
        if event_type == 'arrival':
            stats['arrivals'] += 1
        elif event_type == 'completion' and total_time is not None:
            stats['completions'] += 1
            stats['total_time'] += total_time
            stats['avg_time'] = stats['total_time'] / stats['completions']


    def get_available_server(self, section: str) -> Optional[int]:
        """Find an available server in the given section."""
        try:
            return self.sections[section]["servers"].index(None)
        except ValueError:
            return None
        
    def get_queue_lengths(self):
        """Get current queue lengths for all sections"""
        return {
            section: self.sections[section]["queue"].qsize()
            for section in self.sections
        }

    def handle_arrival(self, time: float):
        """Handle a new customer arrival."""
        # Schedule next arrival
        next_arrival = time + random.expovariate(self.arrival_rate)
        if next_arrival < self.simulation_time:
            heapq.heappush(self.FES, (next_arrival, "arrival", None))

        # Create new client
        new_client = SupermarketClient.generate_random_client(time)
        self.client_counter += 1
        client_id = self.client_counter
        self.active_clients[client_id] = new_client

        # Update metrics
        self.metrics['total_customers'] += 1
        self.metrics['active_customers'] += 1
        self.update_hourly_statistics(time, 'arrival')

        # Get current queue lengths and decide next section
        queue_lengths = self.get_queue_lengths()
        next_section = new_client.decide_next_section(queue_lengths)

        # Schedule section entry
        print(f"Time {time:.2f}: Client {client_id} ({new_client.shopping_type}) arrived, heading to {next_section}")
        heapq.heappush(self.FES, (time, "section_entry", (client_id, next_section)))

    def handle_section_entry(self, time: float, client_id: int, section: str):
        """Handle section entry with proper server and queue management."""
        client = self.active_clients.get(client_id)
        if not client:
            return

        section_data = self.sections[section]
        section_metrics = self.metrics['section_metrics'][section]

        # Add customer to unique customers set
        section_metrics['unique_customers'].add(client_id)

        if section == "other":
            # Special handling for "other" section - immediate service
            server_id = len(section_data["servers"])  # Create new server slot
            section_data["servers"].append(client_id)
            client.enter_section(time, section)

            # Update metrics
            section_metrics['total_arrivals'] += 1
            section_metrics['current_customers'] += 1
            section_metrics['waiting_times'].append(0)  # No waiting time

            # Use the client's remaining time for the section
            service_time = client.remaining_times["other"]
            departure_time = time + service_time
            
            # Schedule departure with the remaining time
            heapq.heappush(self.FES, (departure_time, "section_departure", 
                                    (server_id, section, client_id)))
            print(f"Time {time:.2f}: Client {client_id} ({client.shopping_type}, {client.service_speed}) started service in other section")

        else:
            # Regular handling for other sections
            server_index = self.get_available_server(section)
            
            if server_index is not None:
                # Start service
                section_data["servers"][server_index] = client_id
                client.enter_section(time, section)
                
                # Update metrics
                section_metrics['total_arrivals'] += 1
                section_metrics['current_customers'] += 1
                
                # Use the client's remaining time for the section
                service_time = client.remaining_times[section]
                departure_time = time + service_time
                
                heapq.heappush(self.FES, (departure_time, "section_departure", 
                                        (server_index, section, client_id)))
                
                section_metrics['waiting_times'].append(0)
                print(f"Time {time:.2f}: Client {client_id} ({client.shopping_type}, {client.service_speed}) started service in {section}")
            elif not section_data["queue"].full():
                # Join queue
                section_data["queue"].put(client_id)
                client.enter_section(time, section)
                
                # Update metrics
                section_metrics['total_arrivals'] += 1
                section_metrics['current_customers'] += 1
                section_metrics['queue_lengths'].append(section_data["queue"].qsize())
                
                print(f"Time {time:.2f}: Client {client_id} ({client.shopping_type}, {client.service_speed}) queued in {section}")
            else:
                # Section is full
                section_metrics['rejected_customers'] += 1
                print(f"Time {time:.2f}: Client {client_id} ({client.shopping_type}, {client.service_speed}) rejected from {section} (full)")


                
    def find_alternative_section(self, client, original_section):
        """Find alternative section when original section is full."""
        if original_section in ["butchery", "fresh_food"] and client.remaining_times["other"] > 0:
            return "other"
        return None

    def handle_section_departure(self, time: float, event_data: tuple):
        """Handle client departure from a section with proper remaining time updates."""
        server_index, section, client_id = event_data
        
        client = self.active_clients.get(client_id)
        if not client:
            print(f"Warning: Client {client_id} not found during departure from {section}")
            return

        section_data = self.sections[section]
        section_metrics = self.metrics['section_metrics'][section]

        # Calculate the actual service time
        service_start_time = client.section_metrics[section]["service_start"]
        actual_service_time = time - service_start_time
        
        # Update remaining time for the section to 0 since service is complete
        client.remaining_times[section] = 0

        if section == "other":
            # Special handling for other section with infinite servers
            if server_index < len(section_data["servers"]):
                section_data["servers"][server_index] = None
                while (len(section_data["servers"]) > 0 and 
                    section_data["servers"][-1] is None):
                    section_data["servers"].pop()
            
            # Record service completion
            client.exit_section(time, section)
            section_metrics['service_times'].append(actual_service_time)
            section_metrics['completions'] += 1
            section_metrics['current_customers'] -= 1
            
        else:
            # Regular section handling
            if server_index >= len(section_data["servers"]) or section_data["servers"][server_index] != client_id:
                print(f"Warning: Server state mismatch during departure. Section: {section}, Server: {server_index}")
                return
            
            # Record service completion
            client.exit_section(time, section)
            section_metrics['service_times'].append(actual_service_time)
            section_metrics['completions'] += 1
            section_metrics['current_customers'] -= 1
            
            # Free up the server
            section_data["servers"][server_index] = None
            
            # Process next client from queue if any
            if not section_data["queue"].empty():
                try:
                    next_client_id = section_data["queue"].get_nowait()
                    next_client = self.active_clients.get(next_client_id)
                    
                    if next_client:
                        section_data["servers"][server_index] = next_client_id
                        next_client.section_metrics[section]["service_start"] = time
                        
                        # Use the client's remaining time for the section
                        service_time = next_client.remaining_times[section]
                        departure_time = time + service_time
                        
                        heapq.heappush(self.FES, (departure_time, "section_departure",
                                                (server_index, section, next_client_id)))
                        
                        # Calculate and record waiting time
                        waiting_time = time - next_client.section_metrics[section]["entry_time"]
                        section_metrics['waiting_times'].append(waiting_time)
                        
                        print(f"Time {time:.2f}: Client {next_client_id} ({next_client.shopping_type}, {next_client.service_speed}) started service in {section}")
                except Exception as e:
                    print(f"Error processing next client from queue in {section}: {str(e)}")
        
        # Update queue length metrics
        if section != "other" and hasattr(section_data["queue"], "qsize"):
            section_metrics['queue_lengths'].append(section_data["queue"].qsize())
        
        # Determine next section for departing client
        queue_lengths = {
            sect: (0 if sect == "other" else self.sections[sect]["queue"].qsize())
            for sect in self.sections
        }
        next_section = client.decide_next_section(queue_lengths)
        
        if next_section == "exit":
            self.handle_customer_exit(time, client_id)
        else:
            heapq.heappush(self.FES, (time, "section_entry", (client_id, next_section)))
            print(f"Time {time:.2f}: Client {client_id} ({client.shopping_type}, {client.service_speed}) moving from {section} to {next_section}")
        
        # Update server utilization metrics
        if section != "other":
            servers = section_data["servers"]
            utilization = sum(1 for server in servers if server is not None) / len(servers)
            section_metrics['server_utilization'].append(utilization)
        else:
            section_metrics['server_utilization'].append(1.0)
                
    def check_customer_ready_for_checkout(self, client):
        """Check if customer has completed all necessary shopping."""
        # Check if client has completed their required sections
        if client.shopping_type in ["butch", "both"] and "butchery" not in client.sections_visited:
            return False
        if client.shopping_type in ["fresh", "both"] and "fresh_food" not in client.sections_visited:
            return False
        if "other" not in client.sections_visited:
            return False
        
        # Check if all shopping is complete
        return (client.remaining_times["butchery"] <= 0 and 
                client.remaining_times["fresh_food"] <= 0 and 
                client.remaining_times["other"] <= 0)

    def handle_customer_exit(self, time: float, client_id: int):
        """Handle customer leaving the supermarket."""
        client = self.active_clients.pop(client_id, None)
        if not client:
            print(f"Warning: Client {client_id} not found during final exit")
            return
            
        # Calculate final metrics
        total_time = time - client.arrival_time
        self.metrics['total_time_in_system'].append(total_time)
        self.metrics['completed_customers'] += 1
        self.metrics['active_customers'] -= 1
        
        # Record path
        self.metrics['customer_paths'].append(list(client.sections_visited))
        
        # Record section timing metrics
        for section, timing in client.section_metrics.items():
            if timing["entry_time"] is not None and timing["exit_time"] is not None:
                section_time = timing["exit_time"] - timing["entry_time"]
                if section_time > 0:
                    self.metrics['section_metrics'][section]['total_times'].append(section_time)
        
        # Record journey
        journey = {
            'client_type': client.shopping_type,
            'total_time': total_time,
            'path': list(client.sections_visited),
            'section_times': {
                section: timing["total_time"]
                for section, timing in client.section_metrics.items()
                if timing["total_time"] > 0
            }
        }
        self.metrics['customer_journeys'].append(journey)
        
        # Update hourly statistics
        self.update_hourly_statistics(time, 'completion', total_time)
        
        print(f"Time {time:.2f}: Client {client_id} exited the supermarket. "
              f"Total time: {total_time:.2f}")
    
    def update_hourly_statistics(self, time: float, event_type: str, total_time: float = None):
        """Update hourly statistics with proper averaging."""
        hour = int(time // 60)
        stats = self.metrics['hourly_statistics'][hour]
        
        if event_type == 'arrival':
            stats['arrivals'] += 1
        elif event_type == 'completion' and total_time is not None:
            stats['completions'] += 1
            stats['total_time'] += total_time
            stats['avg_time'] = stats['total_time'] / stats['completions']

    def run(self):
        """Run the supermarket simulation."""
        try:
            while self.FES:
                time, event_type, event_data = heapq.heappop(self.FES)
                if time >= self.simulation_time:
                    break

                if event_type == "arrival":
                    self.handle_arrival(time)
                elif event_type == "section_entry":
                    client_id, section = event_data
                    self.handle_section_entry(time, client_id, section)
                elif event_type == "section_departure":
                    # event_data: (server_index, section, client_id)
                    self.handle_section_departure(time, event_data)

            self.print_summary()
            self.plot_metrics()

        except Exception as e:
            print(f"Error during simulation: {str(e)}")
            raise

    def print_summary(self):
        """Print summary with accurate section statistics."""
        print("\nSupermarket Simulation Summary")
        print("=============================")
        
        print(f"\nOverall Statistics:")
        print(f"Total customers: {self.metrics['total_customers']}")
        print(f"Completed customers: {self.metrics['completed_customers']}")
        print(f"Rejected customers: {self.metrics['rejected_customers']}")
        print(f"Active customers: {self.metrics['active_customers']}")
        
        if self.metrics['total_time_in_system']:
            avg_time = np.mean(self.metrics['total_time_in_system'])
            std_time = np.std(self.metrics['total_time_in_system'])
            print(f"Average time in system: {avg_time:.2f} ± {std_time:.2f} time units")

        print("\nSection Statistics:")
        for section, metrics in self.metrics['section_metrics'].items():
            print(f"\n{section.capitalize()}:")
            print(f"Unique customers served: {len(metrics['unique_customers'])}")
            print(f"Total visits: {metrics['total_arrivals']}")
            print(f"Completions: {metrics['completions']}")
            print(f"Rejected customers: {metrics['rejected_customers']}")
            
            if metrics['service_times']:
                avg_service = np.mean(metrics['service_times'])
                std_service = np.std(metrics['service_times'])
                print(f"Average service time: {avg_service:.2f} ± {std_service:.2f} time units")
            
            if metrics['waiting_times']:
                avg_wait = np.mean(metrics['waiting_times'])
                std_wait = np.std(metrics['waiting_times'])
                print(f"Average waiting time: {avg_wait:.2f} ± {std_wait:.2f} time units")


    def analyze_customer_flow(self):
        """Analyze customer flow through sections."""
        print("\nCustomer Flow Analysis:")
        print("=====================")
        
        # Analyze customer paths
        paths = defaultdict(int)
        for journey in self.metrics['customer_journeys']:
            path = ' -> '.join(journey['path'])
            paths[path] += 1
        
        print("\nCommon customer paths:")
        for path, count in sorted(paths.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"{path}: {count} customers")
        
        # Analyze visit patterns
        print("\nSection visit patterns:")
        for section in ['butchery', 'fresh_food', 'other', 'cashier']:
            metrics = self.metrics['section_metrics'][section]
            unique_customers = len(metrics['unique_customers'])
            total_customers = self.metrics['total_customers']
            if total_customers > 0:
                visit_rate = (unique_customers / total_customers) * 100
                print(f"{section.capitalize()}: {visit_rate:.1f}% of customers visited")
                
    def plot_metrics(self):
        """Plot simulation metrics with proper data handling."""
        plt.figure(figsize=(15, 10))
        
        # 1. Time in System Distribution (top left)
        plt.subplot(2, 2, 1)
        if self.metrics['total_time_in_system']:
            plt.hist(self.metrics['total_time_in_system'], bins=30, color='blue', alpha=0.7)
            plt.title('Distribution of Time in System')
            plt.xlabel('Time Units')
            plt.ylabel('Frequency')
            
            # Add mean and std as text
            mean_time = np.mean(self.metrics['total_time_in_system'])
            std_time = np.std(self.metrics['total_time_in_system'])
            plt.text(0.7, 0.9, f'Mean: {mean_time:.2f}\nStd: {std_time:.2f}', 
                    transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
        else:
            plt.text(0.5, 0.5, 'No completed customers', 
                    horizontalalignment='center', verticalalignment='center')
        
        # 2. Section Statistics (top right)
        plt.subplot(2, 2, 2)
        sections = list(self.sections.keys())
        x = np.arange(len(sections))
        width = 0.25
        
        # Get metrics for each section
        arrivals = [self.metrics['section_metrics'][s]['total_arrivals'] for s in sections]
        completions = [self.metrics['section_metrics'][s]['completions'] for s in sections]
        rejections = [self.metrics['section_metrics'][s]['rejected_customers'] for s in sections]
        
        # Plot grouped bars
        plt.bar(x - width, arrivals, width, label='Arrivals', color='green', alpha=0.7)
        plt.bar(x, completions, width, label='Completions', color='blue', alpha=0.7)
        plt.bar(x + width, rejections, width, label='Rejections', color='red', alpha=0.7)
        
        plt.title('Section Statistics')
        plt.xlabel('Section')
        plt.ylabel('Number of Customers')
        plt.xticks(x, [s.capitalize() for s in sections], rotation=45)
        plt.legend()
        
        # 3. Service Times (bottom left)
        plt.subplot(2, 2, 3)
        service_data = []
        labels = []
        for section in sections:
            times = self.metrics['section_metrics'][section]['service_times']
            if times:
                service_data.append(times)
                labels.append(section.capitalize())
        
        if service_data:
            plt.boxplot(service_data, labels=labels)
            plt.title('Service Times by Section')
            plt.xlabel('Section')
            plt.ylabel('Time Units')
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, 'No service time data', 
                    horizontalalignment='center', verticalalignment='center')
        
        # 4. Customer Types (bottom right)
        plt.subplot(2, 2, 4)
        if self.metrics['customer_journeys']:
            customer_types = defaultdict(int)
            for journey in self.metrics['customer_journeys']:
                customer_types[journey['client_type']] += 1  # Changed from 'type' to 'client_type'
                
            types = list(customer_types.keys())
            counts = list(customer_types.values())
            
            plt.bar(types, counts, color='purple', alpha=0.7)
            plt.title('Customer Type Distribution')
            plt.xlabel('Customer Type')
            plt.ylabel('Number of Customers')
            plt.xticks(rotation=45)
            
            # Add percentage labels
            total = sum(counts)
            if total > 0:
                for i, count in enumerate(counts):
                    percentage = (count/total) * 100
                    plt.text(i, count, f'{percentage:.1f}%', 
                            horizontalalignment='center', verticalalignment='bottom')
        else:
            plt.text(0.5, 0.5, 'No completed customer data', 
                    horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        plt.show()
        
        # Additional plot for hourly statistics if we have data
        if self.metrics['hourly_statistics']:
            plt.figure(figsize=(12, 6))
            hours = sorted(self.metrics['hourly_statistics'].keys())
            arrivals = [self.metrics['hourly_statistics'][h]['arrivals'] for h in hours]
            completions = [self.metrics['hourly_statistics'][h]['completions'] for h in hours]
            
            plt.plot(hours, arrivals, 'g-', label='Arrivals', marker='o')
            plt.plot(hours, completions, 'b-', label='Completions', marker='s')
            plt.title('Hourly Customer Flow')
            plt.xlabel('Hour')
            plt.ylabel('Number of Customers')
            plt.legend()
            plt.grid(True)
            plt.show()