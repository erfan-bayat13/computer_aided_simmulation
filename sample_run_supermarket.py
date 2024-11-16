import random
import heapq
from collections import defaultdict
import numpy as np
from supermarket import SupermarketSimulator
from client_supermarket import SupermarketClient
from Confidence_Interval import ConfidenceIntervalAnalyzer


def test_supermarket(simulation_time=1000, seed=42):
    """
    Run a test simulation of the supermarket and analyze results.
    
    Args:
        simulation_time (float): Duration of simulation
        seed (int): Random seed for reproducibility
        
    Returns:
        dict: Summary of simulation results
    """
    # Create and run simulator
    simulator = SupermarketSimulator(
        butchery_servers=1,
        fresh_food_servers=2,
        cashier_servers=2,
        queue_capacity=7,
        arrival_rate=0.9,
        simulation_time=simulation_time,
        seed=seed
    )
    
    # Run simulation
    simulator.run()
    
    # Analyze results
    results = {
        'total_customers': simulator.metrics['total_customers'],
        'completed_customers': simulator.metrics['completed_customers'],
        'rejected_customers': simulator.metrics['rejected_customers'],
        'avg_time_in_system': 0,
        'customer_type_stats': {},
        'section_stats': {}
    }
    
    # Calculate average time in system
    if simulator.metrics['total_time_in_system']:
        results['avg_time_in_system'] = np.mean(simulator.metrics['total_time_in_system'])
    
    # Analyze customer types
    customer_types = defaultdict(int)
    total_times_by_type = defaultdict(list)
    
    for journey in simulator.metrics['customer_journeys']:
        customer_type = journey['client_type']  # Changed from 'type' to 'client_type'
        customer_types[customer_type] += 1
        total_times_by_type[customer_type].append(journey['total_time'])
    
    # Calculate statistics by customer type
    for ctype in customer_types:
        results['customer_type_stats'][ctype] = {
            'count': customer_types[ctype],
            'avg_time': np.mean(total_times_by_type[ctype]),
            'std_time': np.std(total_times_by_type[ctype])
        }
    
    # Calculate section statistics
    for section, metrics in simulator.metrics['section_metrics'].items():
        results['section_stats'][section] = {
            'arrivals': metrics['total_arrivals'],
            'completions': metrics['completions'],
            'rejections': metrics['rejected_customers'],
            'avg_service_time': np.mean(metrics['service_times']) if metrics['service_times'] else 0,
            'avg_waiting_time': np.mean(metrics['waiting_times']) if metrics['waiting_times'] else 0
        }
    
    # Print detailed analysis
    print("\nDetailed Journey Analysis:")
    print("==========================")
    
    print("\nCustomer Type Statistics:")
    for ctype, stats in results['customer_type_stats'].items():
        print(f"\n{ctype} customers:")
        print(f"  Count: {stats['count']}")
        print(f"  Average time: {stats['avg_time']:.2f} ± {stats['std_time']:.2f} time units")
    
    print("\nSection Performance:")
    for section, stats in results['section_stats'].items():
        print(f"\n{section.capitalize()}:")
        print(f"  Arrivals: {stats['arrivals']}")
        print(f"  Completions: {stats['completions']}")
        print(f"  Rejections: {stats['rejections']}")
        print(f"  Average service time: {stats['avg_service_time']:.2f} time units")
        print(f"  Average waiting time: {stats['avg_waiting_time']:.2f} time units")
    
    # Calculate and print efficiency metrics
    completion_rate = (results['completed_customers'] / results['total_customers'] * 100 
                      if results['total_customers'] > 0 else 0)
    rejection_rate = (results['rejected_customers'] / results['total_customers'] * 100 
                     if results['total_customers'] > 0 else 0)
    
    print("\nEfficiency Metrics:")
    print(f"Completion rate: {completion_rate:.1f}%")
    print(f"Rejection rate: {rejection_rate:.1f}%")
    print(f"Average time in system: {results['avg_time_in_system']:.2f} time units")
    
    return results


# Example usage for running a single simulation arguments can be adjusted in the function:
if __name__ == "__main__":
    print("Test 1: Base scenario")
    print("====================")
    results = test_supermarket(simulation_time=1000, seed=42)


'''
# example usage for running multiple replications and calculating confidence intervals
## please remove the print plots before running the code to not genereate 100  instances of the plots
# Create analyzer instance
analyzer = ConfidenceIntervalAnalyzer(confidence_level=0.95)

base_seed = 42  
seeds = [base_seed + i * 10**3 for i in range(100)]

# Run multiple replications
num_replications = 100  # Generally, 30+ replications give good statistical properties
for i in range(num_replications):
    # Create and run simulation with different random seed
    simulator = SupermarketSimulator(
        seed=seeds[i],  # Use different seed for each replication
        butchery_servers=2,
        fresh_food_servers=2,
        cashier_servers=3,
        queue_capacity=10,
        arrival_rate=0.5,
        simulation_time=1000
    )
    simulator.run()
    
    # Add results to analyzer
    analyzer.add_simulation_run(simulator.metrics)

# Analyze results and print confidence intervals
results = analyzer.analyze_results()
analyzer.print_analysis(results)
'''
