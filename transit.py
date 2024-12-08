from typing import List, Tuple, Dict, Union
import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Union
import matplotlib.pyplot as plt
from scipy.stats import t as student_t
import random
import queue
import heapq
from complete_queue import MultiServerQueueSimulator, Client

class TransientPhaseDetector:
    """Utility class for detecting transient phase in simulation data."""
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize the transient phase detector.
        
        Args:
            confidence_level (float): Confidence level for statistical tests (default: 0.95)
        """
        self.confidence_level = confidence_level
        self.methods = {
            'mser': self.mser_method,
            'batch_means': self.batch_means_method,
            'welch': self.welch_method,
            'running_mean': self.running_mean_method
        }
    
    def detect_transient_phase(self, 
                             time_series: List[float], 
                             methods: List[str] = ['mser', 'batch_means', 'welch'],
                             window_size: int = None) -> Dict[str, Dict]:
        """
        Detect transient phase using multiple methods.
        
        Args:
            time_series: List of observations over time
            methods: List of methods to use ['mser', 'batch_means', 'welch', 'running_mean']
            window_size: Size of moving window (default: len(time_series)//10)
            
        Returns:
            Dictionary containing results from each method
        """
        if window_size is None:
            window_size = max(len(time_series) // 10, 2)
            
        results = {}
        for method in methods:
            if method in self.methods:
                try:
                    result = self.methods[method](time_series, window_size)
                    results[method] = result
                except Exception as e:
                    print(f"Error in {method} method: {str(e)}")
                    results[method] = {'error': str(e)}
            
        return results
    
    def mser_method(self, time_series: List[float], window_size: int) -> Dict:
        """
        Marginal Standard Error Rules (MSER) method for transient detection.
        
        Args:
            time_series: List of observations
            window_size: Moving window size
            
        Returns:
            Dictionary containing cutoff point and statistics
        """
        n = len(time_series)
        min_mser = float('inf')
        cutoff = 0
        
        # Calculate MSER statistic for different truncation points
        mser_values = []
        for d in range(n - window_size):
            truncated = time_series[d:]
            mean = np.mean(truncated)
            mser = np.sum((truncated - mean) ** 2) / (len(truncated) ** 2)
            mser_values.append(mser)
            
            if mser < min_mser:
                min_mser = mser
                cutoff = d
        
        return {
            'cutoff_point': cutoff,
            'mser_statistic': min_mser,
            'mser_values': mser_values
        }
    
    def batch_means_method(self, time_series: List[float], batch_size: int) -> Dict:
        """
        Batch means method for detecting steady state.
        
        Args:
            time_series: List of observations
            batch_size: Size of each batch
            
        Returns:
            Dictionary containing cutoff point and batch statistics
        """
        n = len(time_series)
        
        # Ensure the time series length is divisible by the batch size
        num_batches = n // batch_size
        if num_batches < 2:
            raise ValueError("Batch size is too large or time series is too short for meaningful batch analysis.")
        
        # Calculate batch means
        batch_means = []
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch_mean = np.mean(time_series[start:end])
            batch_means.append(batch_mean)
        
        # Compute the Von Neumann ratio
        diff_squared = np.sum(np.diff(batch_means) ** 2)
        mean_diff_squared = diff_squared / (len(batch_means) - 1)
        variance = np.var(batch_means, ddof=1)  # Using ddof=1 for sample variance
        
        von_neumann_ratio = mean_diff_squared / (2 * variance) if variance != 0 else float('inf')
        
        # Find the cutoff point where the batch means stabilize
        cutoff = self._find_steady_state_batch(batch_means) * batch_size
        
        return {
            'cutoff_point': cutoff,
            'von_neumann_ratio': von_neumann_ratio,
            'batch_means': batch_means
        }

    
    def welch_method(self, time_series: List[float], window_size: int) -> Dict:
        """
        Welch's method for detecting steady state.
        
        Args:
            time_series: List of observations
            window_size: Size of moving window
            
        Returns:
            Dictionary containing cutoff point and test statistics
        """
        n = len(time_series)
        means = []
        variances = []
        
        # Calculate moving averages and variances
        for i in range(0, n - window_size + 1):
            window = time_series[i:i + window_size]
            means.append(np.mean(window))
            variances.append(np.var(window))
        
        # Find where variance stabilizes
        var_changes = np.abs(np.diff(variances))
        cutoff = np.argmin(var_changes) + window_size
        
        return {
            'cutoff_point': cutoff,
            'moving_means': means,
            'moving_variances': variances
        }
    
    def running_mean_method(self, time_series: List[float], window_size: int) -> Dict:
        """
        Running mean method with confidence intervals.
        
        Args:
            time_series: List of observations
            window_size: Size of moving window
            
        Returns:
            Dictionary containing cutoff point and confidence intervals
        """
        n = len(time_series)
        running_means = []
        confidence_intervals = []
        
        for i in range(window_size, n):
            window = time_series[:i]
            mean = np.mean(window)
            std = np.std(window, ddof=1)
            
            # Calculate confidence interval
            t_value = student_t.ppf((1 + self.confidence_level) / 2, df=len(window)-1)
            ci = t_value * (std / np.sqrt(len(window)))
            
            running_means.append(mean)
            confidence_intervals.append(ci)
        
        # Find where confidence intervals stabilize
        ci_changes = np.abs(np.diff(confidence_intervals))
        cutoff = np.argmin(ci_changes) + window_size
        
        return {
            'cutoff_point': cutoff,
            'running_means': running_means,
            'confidence_intervals': confidence_intervals
        }
    
    def _find_steady_state_batch(self, batch_means: List[float]) -> int:
        """Helper method to find steady state batch using confidence intervals."""
        n = len(batch_means)
        for i in range(1, n):
            first_half = batch_means[:i]
            second_half = batch_means[i:]
            
            if len(second_half) < 2:
                continue
                
            t_stat, p_value = stats.ttest_ind(first_half, second_half)
            if p_value > (1 - self.confidence_level):
                return i
        return n // 2
    
    def plot_results(self, time_series: List[float], results: Dict, title: str = "Transient Phase Detection"):
        """
        Plot the results of transient phase detection.
        
        Args:
            time_series: Original time series data
            results: Results from detect_transient_phase
            title: Plot title
        """
        n_methods = len(results)
        fig, axes = plt.subplots(n_methods, 1, figsize=(12, 4*n_methods))
        if n_methods == 1:
            axes = [axes]
        
        time_points = range(len(time_series))
        
        for ax, (method_name, result) in zip(axes, results.items()):
            ax.plot(time_points, time_series, 'b-', alpha=0.5, label='Original Data')
            
            if 'cutoff_point' in result:
                cutoff = result['cutoff_point']
                ax.axvline(x=cutoff, color='r', linestyle='--', 
                          label=f'Cutoff Point ({cutoff})')
                
                # Plot specific method results
                if method_name == 'mser' and 'mser_values' in result:
                    ax2 = ax.twinx()
                    ax2.plot(result['mser_values'], 'g-', alpha=0.5, label='MSER Values')
                    ax2.set_ylabel('MSER Value')
                
                elif method_name == 'welch' and 'moving_means' in result:
                    ax.plot(range(len(result['moving_means'])), 
                           result['moving_means'], 'g-', label='Moving Average')
                
                elif method_name == 'running_mean' and 'running_means' in result:
                    means = result['running_means']
                    cis = result['confidence_intervals']
                    x = range(len(means))
                    ax.plot(x, means, 'g-', label='Running Mean')
                    ax.fill_between(x, 
                                  [m - ci for m, ci in zip(means, cis)],
                                  [m + ci for m, ci in zip(means, cis)],
                                  color='g', alpha=0.2)
            
            ax.set_title(f'{method_name.upper()} Method')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.grid(True)
            ax.legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def analyze_system_stability(self, 
                               metrics: Dict[str, List[float]], 
                               window_sizes: Dict[str, int] = None) -> Dict[str, Dict]:
        """
        Analyze stability of multiple system metrics.
        
        Args:
            metrics: Dictionary of metric names and their time series
            window_sizes: Dictionary of metric names and their window sizes
            
        Returns:
            Dictionary containing analysis results for each metric
        """
        if window_sizes is None:
            window_sizes = {metric: len(values)//10 for metric, values in metrics.items()}
            
        results = {}
        for metric_name, values in metrics.items():
            window_size = window_sizes.get(metric_name, len(values)//10)
            
            # Detect transient phase using all methods
            metric_results = self.detect_transient_phase(
                values, 
                methods=['mser', 'batch_means', 'welch', 'running_mean'],
                window_size=window_size
            )
            
            # Plot results
            self.plot_results(values, metric_results, 
                            title=f"Transient Phase Detection - {metric_name}")
            
            results[metric_name] = metric_results
            
        return results

def initialize_fifo_queue(queue_capacity, initial_queue_size, service_rate):
    """
    Initialize a FIFO queue with a specified number of initial clients.
    
    Args:
        queue_capacity (int): Maximum capacity of the queue
        initial_queue_size (int): Number of clients to start in queue
        service_rate (float): Service rate for generating service times
        
    Returns:
        queue.Queue: Initialized queue with clients
    """
    waiting_queue = queue.Queue(maxsize=queue_capacity)
    
    for _ in range(initial_queue_size):
        service_time = random.expovariate(service_rate)
        initial_client = Client(
            arrival_time=0,
            service_time=service_time,
            client_type="Regular"
        )
        waiting_queue.put(initial_client)
        
    return waiting_queue


class MultiServerQueueSimulator_with_initial_queue(MultiServerQueueSimulator):
    """Extended version of MultiServerQueueSimulator that supports initial queue state."""
    
    def __init__(self, num_servers, queue_capacity, arrival_rate, service_rate, 
                 simulation_time, initial_queue_size=1000):
        """
        Initialize simulator with initial queue state.
        """
        # Call parent class constructor first
        super().__init__(num_servers, queue_capacity, arrival_rate, service_rate, 
                        simulation_time, queue_type="FIFO")
        
        # Initialize the waiting queue with initial clients
        for _ in range(initial_queue_size):
            service_time = random.expovariate(service_rate)
            initial_client = Client(0, service_time, client_type="Regular")
            if not self.waiting_queue.full():
                self.waiting_queue.put(initial_client)
        
        # Update metrics dictionary with initial state and additional transient analysis fields
        self.metrics.update({
            'total_customers': initial_queue_size,
            'active_customers': initial_queue_size,
            'queue_sizes': [initial_queue_size],
            'queue_length_samples': [initial_queue_size],
            'transient_metrics': {
                'queue_sizes_over_time': [(0, initial_queue_size)],
                'server_utilization_over_time': [(0, 0)],
                'running_avg_delays': []
            }
        })
        
        # Try to assign initial clients to servers
        self.assign_initial_clients()

    def assign_initial_clients(self):
        """Attempt to assign initial clients in queue to available servers."""
        for i, server in enumerate(self.servers):
            if server is None and not self.waiting_queue.empty():
                client = self.waiting_queue.get()
                self.servers[i] = client
                client.last_service_start = 0
                
                # Schedule departure
                service_time = random.expovariate(self.service_rate)
                departure_time = service_time
                heapq.heappush(self.FES, (departure_time, "departure", i))
                
                # Update metrics
                current_utilization = sum(1 for s in self.servers if s is not None) / self.num_servers
                self.metrics['transient_metrics']['server_utilization_over_time'].append(
                    (0, current_utilization)
                )

    def update_transient_metrics(self, time):
        """Update metrics used for transient phase detection."""
        current_queue_size = self.waiting_queue.qsize()
        current_utilization = sum(1 for server in self.servers if server is not None) / self.num_servers
        
        # Update transient metrics
        self.metrics['transient_metrics']['queue_sizes_over_time'].append(
            (time, current_queue_size)
        )
        self.metrics['transient_metrics']['server_utilization_over_time'].append(
            (time, current_utilization)
        )
        
        # Update running average delays if we have delay data
        if self.metrics['delays']:
            running_avg = sum(self.metrics['delays']) / len(self.metrics['delays'])
            self.metrics['transient_metrics']['running_avg_delays'].append(
                (time, running_avg)
            )
        
        # Update regular metrics
        self.metrics['queue_sizes'].append(current_queue_size)
        self.metrics['queue_length_samples'].append(current_queue_size)

    def run(self):
        """Run simulation with transient metrics tracking."""
        try:
            while self.FES:
                time, event_type, event_data = heapq.heappop(self.FES)
                if time >= self.simulation_time:
                    break

                if event_type == "arrival":
                    rejected = Client.arrival(
                        time,
                        self.FES,
                        self.waiting_queue,
                        self.arrival_rate,
                        self.service_rate,
                        self.servers,
                        self.metrics
                    )
                    if not rejected:
                        self.update_transient_metrics(time)
                        
                elif event_type == "departure":
                    Client.departure(
                        time,
                        self.FES,
                        self.waiting_queue,
                        event_data,
                        self.servers,
                        self.metrics
                    )
                    self.update_transient_metrics(time)

            # After simulation completes
            self.plot_transient_metrics()
            self.print_summary()

        except Exception as e:
            print(f"Error during simulation: {str(e)}")
            raise

    def plot_transient_metrics(self):
        """Plot metrics for transient analysis."""
        plt.figure(figsize=(15, 10))
        
        # Queue Size Over Time
        plt.subplot(3, 2, 1)
        times, sizes = zip(*self.metrics['transient_metrics']['queue_sizes_over_time'])
        plt.plot(times, sizes, 'b-', label='Queue Size')
        plt.title('Queue Size Over Time')
        plt.xlabel('Time')
        plt.ylabel('Queue Size')
        plt.grid(True)
        plt.legend()
        
        # Server Utilization
        plt.subplot(3, 2, 2)
        times, utils = zip(*self.metrics['transient_metrics']['server_utilization_over_time'])
        plt.plot(times, utils, 'g-', label='Server Utilization')
        plt.title('Server Utilization Over Time')
        plt.xlabel('Time')
        plt.ylabel('Utilization')
        plt.grid(True)
        plt.legend()
        
        # Average Delay
        plt.subplot(3, 2, 3)
        if self.metrics['delays']:
            cumulative_avg = [np.mean(self.metrics['delays'][:i+1]) 
                            for i in range(len(self.metrics['delays']))]
            plt.plot(cumulative_avg, 'r-', label='Running Average Delay')
            plt.title('Running Average Delay Over Time')
            plt.xlabel('Completed Customers')
            plt.ylabel('Average Delay')
            plt.grid(True)
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No delay data available', 
                    horizontalalignment='center', verticalalignment='center')
            
        # Raw Delays
        plt.subplot(3, 2, 4)
        if self.metrics['delays']:
            plt.plot(self.metrics['delays'], 'r-', label='Customer Delays')
            plt.title('Customer Delays Over Time')
            plt.xlabel('Customer Index')
            plt.ylabel('Delay Time')
            plt.grid(True)
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No delay data available', 
                    horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        plt.show()

    def print_summary(self):
        """Print detailed simulation summary."""
        print("\nSimulation Summary")
        print("=================")
        
        print(f"\nOverall Statistics:")
        print(f"Total customers: {self.metrics['total_customers']}")
        print(f"Completed customers: {self.metrics['total_customers_served']}")
        print(f"Rejected customers: {self.metrics['rejected_customers']}")
        print(f"Active customers: {self.metrics['active_customers']}")
        
        if self.metrics['delays']:
            avg_delay = np.mean(self.metrics['delays'])
            std_delay = np.std(self.metrics['delays'])
            print(f"Average delay: {avg_delay:.2f} ± {std_delay:.2f} time units")
        
        # Calculate and print server utilization
        final_utilization = self.metrics['transient_metrics']['server_utilization_over_time'][-1][1]
        avg_utilization = np.mean([util for _, util in 
                                 self.metrics['transient_metrics']['server_utilization_over_time']])
        print(f"\nServer Utilization:")
        print(f"Final utilization: {final_utilization:.2f}")
        print(f"Average utilization: {avg_utilization:.2f}")
        
        # Queue statistics
        if self.metrics['queue_sizes']:
            avg_queue = np.mean(self.metrics['queue_sizes'])
            max_queue = max(self.metrics['queue_sizes'])
            print(f"\nQueue Statistics:")
            print(f"Average queue size: {avg_queue:.2f}")
            print(f"Maximum queue size: {max_queue}")

# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create simulator with initial queue
    simulator = MultiServerQueueSimulator_with_initial_queue(
        num_servers=1,
        queue_capacity=100,
        arrival_rate=1.2,
        service_rate=1.0,
        simulation_time=10000,
        initial_queue_size=0
    )
    
    # Run the simulation
    simulator.run()