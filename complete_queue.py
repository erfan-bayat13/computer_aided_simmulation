import random
import queue
import heapq
import numpy as np
import matplotlib.pyplot as plt

class Client:
    def __init__(self, arrival_time, service_time, client_type="Regular"):
        self.client_type = client_type
        self.arrival_time = arrival_time
        self.service_time = service_time
        self.priority = {"VIP": 3, "Regular": 2, "Economy": 1}[client_type]
        self.remaining_service_time = service_time
        self.last_service_start = None
        self.preempted = False

    @staticmethod
    def arrival(time, FES, waiting_queues, arrival_rate, service_rate, servers, metrics, is_preemptive=False):
        # Schedule next arrival
        inter_arrival_time = random.expovariate(arrival_rate)
        next_arrival_time = time + inter_arrival_time
        heapq.heappush(FES, (next_arrival_time, "arrival", None))

        # Create new client
        client_type = random.choices(["VIP", "Regular", "Economy"], weights=[0.2, 0.5, 0.3])[0]
        service_time = random.expovariate(service_rate)
        new_client = Client(time, service_time, client_type=client_type)
        priority = new_client.priority

        # Record queue length sample
        current_length = sum(q.qsize() for q in waiting_queues.values()) if isinstance(waiting_queues, dict) else waiting_queues.qsize()
        metrics['queue_length_samples'].append(current_length)
        metrics['queue_length_timestamps'].append(time)

        # Handle priority queues
        if isinstance(waiting_queues, dict):
            # Handle preemption for VIP clients
            if is_preemptive and client_type == "VIP":
                for i, server in enumerate(servers):
                    if server is not None and server.priority < 3:
                        preempted_client = server
                        time_served = time - preempted_client.last_service_start
                        preempted_client.remaining_service_time = max(0, preempted_client.remaining_service_time - time_served)
                        preempted_client.preempted = True
                        preempted_client.last_service_start = None

                        # Remove existing departure event
                        FES = [event for event in FES if not (event[1] == "departure" and event[2] == i)]
                        heapq.heapify(FES)

                        # Put VIP client in service
                        servers[i] = new_client
                        new_client.last_service_start = time
                        metrics['priority_metrics'][priority]['arrived'] += 1

                        # Schedule departure for VIP client
                        departure_time = time + new_client.service_time
                        heapq.heappush(FES, (departure_time, "departure", i))

                        if not waiting_queues[preempted_client.priority].full():
                            waiting_queues[preempted_client.priority].put(preempted_client)
                            metrics['preemptions'] += 1
                            return False
                        metrics['rejected_customers'] += 1
                        return True

            if not waiting_queues[priority].full():
                waiting_queues[priority].put(new_client)
                metrics['queue_sizes'].append(sum(q.qsize() for q in waiting_queues.values()))
                metrics['priority_metrics'][priority]['arrived'] += 1
                metrics['priority_metrics'][priority]['queue_sizes'].append(waiting_queues[priority].qsize())
            else:
                metrics['rejected_customers'] += 1
                return True
        else:
            if not waiting_queues.full():
                waiting_queues.put(new_client)
                metrics['queue_sizes'].append(waiting_queues.qsize())
            else:
                metrics['rejected_customers'] += 1
                return True

        Client.assign_clients_to_servers(time, FES, waiting_queues, servers, service_rate, metrics)
        return False

    @staticmethod
    def assign_clients_to_servers(time, FES, waiting_queues, servers, service_rate, metrics):
        for i, server in enumerate(servers):
            if server is None:
                next_client = None
                
                # Get next client based on queue type
                if isinstance(waiting_queues, dict):
                    next_client = Client.get_highest_priority_client(waiting_queues)
                else:
                    if not waiting_queues.empty():
                        next_client = waiting_queues.get()

                if next_client:
                    servers[i] = next_client
                    next_client.last_service_start = time
                    service_time = (next_client.remaining_service_time 
                                  if next_client.preempted 
                                  else next_client.service_time)
                    departure_time = time + service_time
                    heapq.heappush(FES, (departure_time, "departure", i))
                    waiting_time = time - next_client.arrival_time
                    metrics['delays'].append(waiting_time)

    @staticmethod
    def get_highest_priority_client(waiting_queues):
        for priority in sorted(waiting_queues.keys(), reverse=True):
            if not waiting_queues[priority].empty():
                return waiting_queues[priority].get()
        return None

    @staticmethod
    def departure(time, FES, waiting_queues, server_index, servers, metrics):
        departed_client = servers[server_index]
        if departed_client is None:
            return

        servers[server_index] = None
        total_delay = time - departed_client.arrival_time
        metrics['delays'].append(total_delay)
        metrics['total_customers_served'] += 1

        # Record queue length sample
        current_length = sum(q.qsize() for q in waiting_queues.values()) if isinstance(waiting_queues, dict) else waiting_queues.qsize()
        metrics['queue_length_samples'].append(current_length)
        metrics['queue_length_timestamps'].append(time)

        if isinstance(waiting_queues, dict):
            priority = departed_client.priority
            metrics['priority_metrics'][priority]['served'] += 1
            metrics['priority_metrics'][priority]['delays'].append(total_delay)

        next_client = None
        if isinstance(waiting_queues, dict):
            next_client = Client.get_highest_priority_client(waiting_queues)
        else:
            if not waiting_queues.empty():
                next_client = waiting_queues.get()

        if next_client:
            servers[server_index] = next_client
            next_client.last_service_start = time
            service_time = next_client.remaining_service_time if next_client.preempted else next_client.service_time
            departure_time = time + service_time
            heapq.heappush(FES, (departure_time, "departure", server_index))

class MultiServerQueueSimulator:
    def __init__(self, num_servers, queue_capacity, arrival_rate, service_rate, 
                 simulation_time, queue_type="FIFO", is_preemptive=False):
        self.num_servers = num_servers
        self.queue_capacity = queue_capacity
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.simulation_time = simulation_time
        self.queue_type = queue_type
        self.is_preemptive = is_preemptive
        self.servers = [None] * num_servers

        if self.queue_type == "FIFO":
            self.waiting_queue = queue.Queue(maxsize=queue_capacity)
        elif self.queue_type == "LIFO":
            self.waiting_queue = queue.LifoQueue(maxsize=queue_capacity)
        elif self.queue_type == "Priority" or self.is_preemptive:
            self.waiting_queue = {
                3: queue.Queue(maxsize=queue_capacity),
                2: queue.Queue(maxsize=queue_capacity),
                1: queue.Queue(maxsize=queue_capacity)
            }
        else:
            raise ValueError(f"Invalid queue type specified: {self.queue_type}")

        self.metrics = {
            'total_customers_arrived': 0,
            'total_customers_served': 0,
            'rejected_customers': 0,
            'delays': [],
            'preemptions': 0,
            'queue_sizes': [],
            'total_time_in_system': [],
            'queue_length_samples': [],
            'queue_length_timestamps': [],
            'last_queue_sample_time': 0,
            'priority_metrics': {
                3: {'arrived': 0, 'served': 0, 'delays': [], 'queue_sizes': []},
                2: {'arrived': 0, 'served': 0, 'delays': [], 'queue_sizes': []},
                1: {'arrived': 0, 'served': 0, 'delays': [], 'queue_sizes': []}
            }
        }

        self.FES = []
        heapq.heappush(self.FES, (0, "arrival", None))

    def calculate_time_weighted_queue_length(self):
        if not self.metrics['queue_length_timestamps']:
            return 0
            
        total_area = 0
        for i in range(1, len(self.metrics['queue_length_timestamps'])):
            time_diff = self.metrics['queue_length_timestamps'][i] - self.metrics['queue_length_timestamps'][i-1]
            total_area += self.metrics['queue_length_samples'][i-1] * time_diff
            
        total_time = self.metrics['queue_length_timestamps'][-1] - self.metrics['queue_length_timestamps'][0]
        return total_area / total_time if total_time > 0 else 0

    def run(self):
        current_time = 0

        try:
            while self.FES:
                event = heapq.heappop(self.FES)
                current_time = event[0]
                event_type = event[1]

                if current_time >= self.simulation_time:
                    break

                if event_type == "arrival":
                    self.metrics['total_customers_arrived'] += 1
                    rejected = Client.arrival(
                        current_time,
                        self.FES,
                        self.waiting_queue,
                        self.arrival_rate,
                        self.service_rate,
                        self.servers,
                        self.metrics,
                        is_preemptive=self.is_preemptive
                    )

                elif event_type == "departure":
                    server_index = event[2]
                    Client.departure(
                        current_time,
                        self.FES,
                        self.waiting_queue,
                        server_index,
                        self.servers,
                        self.metrics
                    )

        except Exception as e:
            print(f"Error during simulation: {str(e)}")
            raise

        if isinstance(self.waiting_queue, dict):
            remaining_in_queue = sum(q.qsize() for q in self.waiting_queue.values())
        else:
            remaining_in_queue = self.waiting_queue.qsize()
            
        remaining_in_servers = sum(1 for server in self.servers if server is not None)
        self.metrics['remaining_customers'] = remaining_in_queue + remaining_in_servers

        self.print_summary()
        self.plot_metrics()

    def print_summary(self):
        print("\nSimulation Summary")
        print("=================")
        
        queue_type_str = f"{'Preemptive ' if self.is_preemptive else ''}{self.queue_type} Queue"
        print(f"Queue Type: {queue_type_str}")
        print(f"Number of servers: {self.num_servers}")
        print(f"Queue capacity: {self.queue_capacity}")
        
        print(f"\nCustomer Statistics:")
        print(f"Total customers arrived: {self.metrics['total_customers_arrived']}")
        print(f"Total customers served: {self.metrics['total_customers_served']}")
        print(f"Total customers rejected: {self.metrics['rejected_customers']}")
        print(f"Remaining customers: {self.metrics['remaining_customers']}")
        
        # New metrics
        avg_queue_length = self.calculate_time_weighted_queue_length()
        print(f"\nQueue Performance Metrics:")
        print(f"Average Queue Length: {avg_queue_length:.2f}")
        
        if self.metrics['delays']:
            avg_waiting_time = np.mean(self.metrics['delays'])
            print(f"Average Waiting Time in Queue: {avg_waiting_time:.2f} time units")
            
        if self.metrics['total_time_in_system']:
            avg_time_in_system = np.mean(self.metrics['total_time_in_system'])
            print(f"Average Time in System: {avg_time_in_system:.2f} time units")
        
        if self.is_preemptive:
            print(f"\nPreemption Statistics:")
            print(f"Total preemptions: {self.metrics['preemptions']}")

        if self.queue_type == "Priority" or self.is_preemptive:
            print("\nPriority-Specific Statistics:")
            for priority, name in [(3, "VIP"), (2, "Regular"), (1, "Economy")]:
                metrics = self.metrics['priority_metrics'][priority]
                print(f"\n{name} Customers:")
                print(f"  Arrived: {metrics['arrived']}")
                print(f"  Served: {metrics['served']}")
                if metrics['delays']:
                    avg_delay = np.mean(metrics['delays'])
                    print(f"  Average delay: {avg_delay:.2f} time units")

    def plot_metrics(self):
        plt.figure(figsize=(15, 10))

        # 1. Queue Size Distribution
        plt.subplot(2, 2, 1)
        if self.metrics['queue_sizes']:
            plt.hist(self.metrics['queue_sizes'], bins=30, color='blue', alpha=0.7)
            plt.title('Queue Size Distribution')
            plt.xlabel('Queue Size')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)

        # 2. Delay Distribution
        plt.subplot(2, 2, 2)
        if self.metrics['delays']:
            plt.hist(self.metrics['delays'], bins=30, color='green', alpha=0.7)
            plt.title('Customer Delay Distribution')
            plt.xlabel('Delay Time')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)

        # 3. Customer Type Distribution
        plt.subplot(2, 2, 3)
        if self.queue_type == "Priority" or self.is_preemptive:
            types = ['VIP', 'Regular', 'Economy']
            arrived = [self.metrics['priority_metrics'][i+1]['arrived'] for i in range(3)]
            served = [self.metrics['priority_metrics'][i+1]['served'] for i in range(3)]
            
            x = np.arange(len(types))
            width = 0.35
            
            plt.bar(x - width/2, arrived, width, label='Arrived', color='blue', alpha=0.7)
            plt.bar(x + width/2, served, width, label='Served', color='green', alpha=0.7)
            plt.title('Customer Types Distribution')
            plt.xlabel('Customer Type')
            plt.ylabel('Number of Customers')
            plt.xticks(x, types)
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.title('Customer Distribution\n(Not applicable for FIFO/LIFO queues)')

        # 4. System Time Distribution
        plt.subplot(2, 2, 4)
        if self.metrics['total_time_in_system']:
            plt.hist(self.metrics['total_time_in_system'], bins=30, color='red', alpha=0.7)
            plt.title('Time in System Distribution')
            plt.xlabel('Time in System')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)

            # Add mean and std as text
            mean_time = np.mean(self.metrics['total_time_in_system'])
            std_time = np.std(self.metrics['total_time_in_system'])
            plt.text(0.7, 0.9, f'Mean: {mean_time:.2f}\nStd: {std_time:.2f}', 
                    transform=plt.gca().transAxes, 
                    bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.show()

        # Additional plot for time series data
        if self.metrics['queue_length_timestamps']:
            plt.figure(figsize=(12, 6))
            plt.plot(self.metrics['queue_length_timestamps'], 
                    self.metrics['queue_length_samples'], 
                    'b-', alpha=0.6)
            plt.title('Queue Length Over Time')
            plt.xlabel('Time')
            plt.ylabel('Queue Length')
            plt.grid(True, alpha=0.3)
            
            # Add average line
            avg_queue_length = self.calculate_time_weighted_queue_length()
            plt.axhline(y=avg_queue_length, color='r', linestyle='--', 
                       label=f'Average: {avg_queue_length:.2f}')
            plt.legend()
            plt.show()