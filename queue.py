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

    @staticmethod
    def arrival(time, FES, waiting_queues, arrival_rate, service_rate, num_servers, servers, metrics):
        # Schedule the next arrival
        inter_arrival_time = random.expovariate(arrival_rate)
        next_arrival_time = time + inter_arrival_time
        heapq.heappush(FES, (next_arrival_time, "arrival"))

        # Determine the type of the client and assign priority
        client_type = random.choices(["VIP", "Regular", "Economy"], weights=[0.2, 0.5, 0.3])[0]
        priority = {"VIP": 3, "Regular": 2, "Economy": 1}[client_type]

        # Create a new client instance
        service_time = random.expovariate(service_rate)
        new_client = Client(time, service_time, client_type=client_type)

        rejected = False
        # Handle priority queue
        if isinstance(waiting_queues, dict):
            if not waiting_queues[priority].full():
                waiting_queues[priority].put(new_client)
                #print(f"{client_type} client arrived at {time:.2f} and joined the priority queue.")
                metrics['queue_sizes'].append(sum(q.qsize() for q in waiting_queues.values()))
                metrics['priority_metrics'][priority]['arrived'] += 1
                metrics['priority_metrics'][priority]['queue_sizes'].append(waiting_queues[priority].qsize())
            else:
                # Queue is full, client is rejected
                #print(f"{client_type} client arrived at {time:.2f} but was rejected (queue full).")
                rejected = True
                #metrics['rejected_customers'] += 1
                metrics['queue_sizes'].append(sum(q.qsize() for q in waiting_queues.values()))
        else:
            # Handle FIFO and LIFO queues
            if not waiting_queues.full():
                waiting_queues.put(new_client)
                #print(f"{client_type} client arrived at {time:.2f} and joined the queue.")
                metrics['queue_sizes'].append(waiting_queues.qsize())
            else:
                # Queue is full, client is rejected
                #print(f"{client_type} client arrived at {time:.2f} but was rejected (queue full).")
                rejected = True
                #metrics['rejected_customers'] += 1
                metrics['queue_sizes'].append(waiting_queues.qsize())
        # Try to assign clients to available servers
        Client.assign_clients_to_servers(time, FES, waiting_queues, servers, service_rate, metrics)
        return rejected
    

    @staticmethod
    def assign_clients_to_servers(time, FES, waiting_queues, servers, service_rate, metrics):
        for i, server in enumerate(servers):
            if server is None:
                highest_priority_client = Client.get_highest_priority_client(waiting_queues)
                if highest_priority_client:
                    servers[i] = highest_priority_client
                    service_time = random.expovariate(service_rate)
                    departure_time = time + service_time
                    heapq.heappush(FES, (departure_time, "departure", i))
                    waiting_time = time - highest_priority_client.arrival_time
                    metrics['delays'].append(waiting_time)

    @staticmethod
    def get_highest_priority_client(waiting_queues):
        if isinstance(waiting_queues, dict):
            for priority in sorted(waiting_queues.keys(), reverse=True):
                if not waiting_queues[priority].empty():
                    return waiting_queues[priority].get()
        else:
            # If using a single priority queue
            return waiting_queues.get() if not waiting_queues.empty() else None
        return None

    @staticmethod
    def departure(time, FES, waiting_queues, server_index, servers, metrics):
        # Depart the client from the specified server
        departed_client = servers[server_index]
        servers[server_index] = None
        total_delay = time - departed_client.arrival_time
        metrics['delays'].append(total_delay)

        if isinstance(waiting_queues, dict):
            priority = {"VIP": 3, "Regular": 2, "Economy": 1}[departed_client.client_type]
            metrics['priority_metrics'][priority]['served'] += 1
            metrics['priority_metrics'][priority]['delays'].append(total_delay)

        # Check if there's a client waiting in the queues, starting from the highest priority
        if isinstance(waiting_queues, dict):
            for priority in sorted(waiting_queues.keys(), reverse=True):  # Iterate from VIP to Economy
                if not waiting_queues[priority].empty():
                    next_client = waiting_queues[priority].get()
                    servers[server_index] = next_client
                    departure_time = time + next_client.service_time
                    heapq.heappush(FES, (departure_time, "departure", server_index))
                    #print(f"{next_client.client_type} client from priority queue is now being served by server {server_index} at {time:.2f}.")
                    break
        else:
            if not waiting_queues.empty():
                next_client = waiting_queues.get()
                servers[server_index] = next_client
                departure_time = time + next_client.service_time
                heapq.heappush(FES, (departure_time, "departure", server_index))
                #print(f"{next_client.client_type} client from queue is now being served by server {server_index} at {time:.2f}.")
        



class MultiServerQueueSimulator:
    def __init__(self, num_servers, queue_capacity, arrival_rate, service_rate, simulation_time, queue_type="FIFO"):
        self.num_servers = num_servers
        self.queue_capacity = queue_capacity
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.simulation_time = simulation_time

        self.servers = [None] * num_servers  # None means the server is idle
        if queue_type == "FIFO":
            self.waiting_queue = queue.Queue(maxsize=queue_capacity)
        elif queue_type == "LIFO":
            self.waiting_queue = queue.LifoQueue(maxsize=queue_capacity)
        elif queue_type == "Priority":
            # Create a priority-based waiting queue using a hashmap
            self.waiting_queue = {
                3: queue.Queue(maxsize=queue_capacity),  # VIP queue
                2: queue.Queue(maxsize=queue_capacity),  # Regular queue
                1: queue.Queue(maxsize=queue_capacity)   # Economy queue
            }
        else:
            raise ValueError("Invalid queue type specified.")
        self.FES = []  # Future Event Set, represented as a priority queue (heap)
        heapq.heappush(self.FES, (0, "arrival"))  # Initial event to start arrivals
        
        # Metrics
        self.metrics = {
            'total_customers_arrived': 0,
            'total_customers_served': 0,
            'rejected_customers': 0,
            'delays': [],
            'queue_sizes': [],
            'priority_metrics': {
                3: {'arrived': 0, 'served': 0, 'delays': [], 'queue_sizes': []},  # VIP
                2: {'arrived': 0, 'served': 0, 'delays': [], 'queue_sizes': []},  # Regular
                1: {'arrived': 0, 'served': 0, 'delays': [], 'queue_sizes': []}   # Economy
            }
        }

    def run(self):
        current_time = 0

        while current_time < self.simulation_time and self.FES:
            try:
                # Get the next event
                next_event = heapq.heappop(self.FES)
                next_event_time = next_event[0]
                event_type = next_event[1]
                current_time = next_event_time

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
                        self.num_servers,
                        self.servers,
                        self.metrics
                    )
                    if rejected:
                        self.metrics['rejected_customers'] += 1

                elif event_type == "departure":
                    server_index = next_event[2]
                    Client.departure(
                        current_time,
                        self.FES,
                        self.waiting_queue,
                        server_index,
                        self.servers,
                        self.metrics
                    )
                    self.metrics['total_customers_served'] += 1
            except Exception as e:
                print(f"Error during simulation run: {e}")

        # Count remaining customers in queues and servers
        remaining_in_queue = sum(q.qsize() for q in self.waiting_queue.values()) if isinstance(self.waiting_queue, dict) else self.waiting_queue.qsize()
        remaining_in_servers = sum(1 for server in self.servers if server is not None)
        self.metrics['remaining_customers'] = remaining_in_queue + remaining_in_servers

        # Print summary of results
        self.print_summary()
        self.plot_metrics()

    def print_summary(self):
        total_customers = self.metrics['total_customers_arrived']
        served_customers = self.metrics['total_customers_served']
        rejected_customers = self.metrics['rejected_customers']
        remaining_customers = self.metrics['remaining_customers']
        delays = self.metrics['delays']

        avg_delay = np.mean(delays) if delays else 0
        std_dev_delay = np.std(delays) if delays else 0
        avg_queue_size = np.mean(self.metrics['queue_sizes']) if self.metrics['queue_sizes'] else 0
        loss_probability = rejected_customers / total_customers if total_customers > 0 else 0

        # Validate if counts add up
        calculated_total = served_customers + rejected_customers + remaining_customers
        if calculated_total != total_customers:
            print(f"\nWarning: The total number of customers ({total_customers}) does not match "
                  f"the sum of served ({served_customers}), rejected ({rejected_customers}), "
                  f"and remaining ({remaining_customers}) customers ({calculated_total}).")
        else:
            print("\nCustomer counts validated successfully.")

        print("\nSimulation Summary:")
        print(f"Total customers arrived: {total_customers}")
        print(f"Total customers served: {served_customers}")
        print(f"Total customers rejected: {rejected_customers}")
        print(f"Average delay: {avg_delay:.2f} time units")
        print(f"Standard deviation of delay: {std_dev_delay:.2f} time units")
        print(f"Average queue size: {avg_queue_size:.2f}")
        print(f"Remaining customers in queue: {remaining_customers}")
        print(f"Loss/dropping probability: {loss_probability:.2f}")

        # Validate if counts add up
        calculated_total = served_customers + rejected_customers + remaining_customers
        if calculated_total != total_customers:
            print(f"\nWarning: The total number of customers ({total_customers}) does not match "
                    f"the sum of served ({served_customers}), rejected ({rejected_customers}), "
                    f"and remaining ({remaining_customers}) customers ({calculated_total}).")

        if isinstance(self.waiting_queue, dict):
            for priority, metrics in self.metrics['priority_metrics'].items():
                arrived = metrics['arrived']
                served = metrics['served']
                delays = metrics['delays']
                avg_delay = np.mean(delays) if delays else 0
                std_dev_delay = np.std(delays) if delays else 0

                priority_name = {3: "VIP", 2: "Regular", 1: "Economy"}[priority]
                print(f"\n{priority_name} Queue Summary:")
                print(f"Total customers arrived: {arrived}")
                print(f"Total customers served: {served}")
                print(f"Average delay: {avg_delay:.2f} time units")
                print(f"Standard deviation of delay: {std_dev_delay:.2f} time units")

    def plot_metrics(self):
        delays = self.metrics['delays']
        queue_sizes = self.metrics['queue_sizes']

        # Plot histogram of delays
        if delays:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.hist(delays, bins=20, color='blue', alpha=0.7)
            plt.xlabel('Delay (time units)')
            plt.ylabel('Frequency')
            plt.title('Histogram of Delays')

        # Plot queue size distribution over time
        if queue_sizes:
            plt.subplot(1, 2, 2)
            plt.hist(queue_sizes, bins=range(0, max(queue_sizes) + 2), color='green', alpha=0.7, align='left')
            plt.xlabel('Queue Size')
            plt.ylabel('Frequency')
            plt.title('Distribution of Queue Sizes')

        plt.tight_layout()
        plt.show()

        # Plot metrics for each priority queue
        if isinstance(self.waiting_queue, dict):
            for priority, metrics in self.metrics['priority_metrics'].items():
                delays = metrics['delays']
                queue_sizes = metrics['queue_sizes']

                priority_name = {3: "VIP", 2: "Regular", 1: "Economy"}[priority]

                # Plot histogram of delays for each priority queue
                if delays:
                    plt.figure(figsize=(12, 5))
                    plt.subplot(1, 2, 1)
                    plt.hist(delays, bins=20, color='blue', alpha=0.7)
                    plt.xlabel('Delay (time units)')
                    plt.ylabel('Frequency')
                    plt.title(f'Histogram of Delays for {priority_name} Queue')

                # Plot queue size distribution over time for each priority queue
                if queue_sizes:
                    plt.subplot(1, 2, 2)
                    plt.hist(queue_sizes, bins=range(0, max(queue_sizes) + 2), color='green', alpha=0.7, align='left')
                    plt.xlabel('Queue Size')
                    plt.ylabel('Frequency')
                    plt.title(f'Distribution of Queue Sizes for {priority_name} Queue')

                plt.tight_layout()
                plt.show()

# Parameters for simulation
num_servers = 3
queue_capacity = 5
arrival_rate = 0.9  # Average of 1 customer every ~1.11 time units
service_rate = 0.6  # Average of 1 service every ~1.67 time units
simulation_time = 1000  # Total time to run the simulation

# Running the simulation
simulator = MultiServerQueueSimulator(num_servers, queue_capacity, arrival_rate, service_rate, simulation_time, queue_type="LIFO")
simulator.run()