import networkx as nx
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import memory_profiler

# Implementation 1: NetworkX's built-in method (baseline)
def generate_gnp_baseline(n, p):
    return nx.fast_gnp_random_graph(n, p)

# Implementation 2: Set-based method
def generate_gnp_linear_set(n, p):
    # Generate number of edges using Poisson approximation
    lambda_param = p * n * (n-1) / 2
    m = np.random.poisson(lambda_param)
    
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    edges = set()
    while len(edges) < m:
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i != j:
            edge = tuple(sorted((i, j)))
            edges.add(edge)
    
    G.add_edges_from(edges)
    return G

# Implementation 3: Dictionary-based method
def generate_gnp_linear_dict(n, p):
    lambda_param = p * n * (n-1) / 2
    m = np.random.poisson(lambda_param)
    edges = {}
    edge_count = 0
    
    while edge_count < m:
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i != j:
            edge_hash = ((i + j) * (i + j + 1)) // 2 + j
            if edge_hash not in edges:
                edges[edge_hash] = (i, j)
                edge_count += 1
    
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges.values())
    return G

# Implementation 4: Optimized batch method
def generate_gnp_linear_optimized(n, p):
    lambda_param = p * n * (n-1) / 2
    m = np.random.poisson(lambda_param)
    
    edges = np.zeros((m, 2), dtype=np.int32)
    edge_set = set()
    
    batch_size = min(1000, m)
    current_edges = 0
    
    while current_edges < m:
        batch_i = np.random.randint(0, n, batch_size)
        batch_j = np.random.randint(0, n, batch_size)
        
        for i, j in zip(batch_i, batch_j):
            if i != j:
                edge = tuple(sorted((i, j)))
                if edge not in edge_set:
                    edge_set.add(edge)
                    edges[current_edges] = [i, j]
                    current_edges += 1
                    if current_edges == m:
                        break
    
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges[:current_edges])
    return G

def benchmark_time(func, n, p, num_trials=5):
    """Measure execution time for a given implementation"""
    times = []
    for _ in range(num_trials):
        start_time = time.time()
        G = func(n, p)
        end_time = time.time()
        times.append(end_time - start_time)
    return np.mean(times), np.std(times)

def benchmark_memory(func, n, p):
    """Measure peak memory usage for a given implementation"""
    @memory_profiler.profile
    def wrapper():
        return func(n, p)
    
    # Capture memory profiler output
    import io
    import sys
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    
    wrapper()
    
    sys.stdout = old_stdout
    memory_usage = max(float(line.split()[3]) for line in new_stdout.getvalue().split('\n')[1:-1] if line.strip())
    return memory_usage

def run_comprehensive_benchmark():
    """Run comprehensive benchmarks across different n values"""
    implementations = {
        'NetworkX Baseline': generate_gnp_baseline,
        'Set-based': generate_gnp_linear_set,
        'Dictionary-based': generate_gnp_linear_dict,
        'Optimized Batch': generate_gnp_linear_optimized
    }
    
    n_values = [1000, 5000, 10000, 50000]
    results = {name: {'time': [], 'time_std': [], 'memory': []} for name in implementations}
    
    for n in tqdm(n_values, desc="Benchmarking across n values"):
        p = 3.0 / n  # Setting p = O(1/n)
        
        for name, func in implementations.items():
            # Time benchmark
            mean_time, std_time = benchmark_time(func, n, p)
            results[name]['time'].append(mean_time)
            results[name]['time_std'].append(std_time)
            
            # Memory benchmark
            mem_usage = benchmark_memory(func, n, p)
            results[name]['memory'].append(mem_usage)
    
    # Create visualization
    plt.figure(figsize=(15, 6))
    
    # Time plot
    plt.subplot(1, 2, 1)
    for name in implementations:
        plt.errorbar(n_values, results[name]['time'], 
                    yerr=results[name]['time_std'],
                    label=name, marker='o')
    plt.xlabel('Number of nodes (n)')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Time Comparison')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    
    # Memory plot
    plt.subplot(1, 2, 2)
    for name in implementations:
        plt.plot(n_values, results[name]['memory'], 
                label=name, marker='o')
    plt.xlabel('Number of nodes (n)')
    plt.ylabel('Peak Memory Usage (MB)')
    plt.title('Memory Usage Comparison')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    
    plt.tight_layout()
    return plt, results

if __name__ == "__main__":
    plt, results = run_comprehensive_benchmark()
    plt.show()
    
    # Print detailed results
    print("\nDetailed Benchmark Results:")
    print("-" * 50)
    for n in [1000, 5000, 10000, 50000]:
        print(f"\nResults for n = {n}:")
        for name in results:
            idx = [1000, 5000, 10000, 50000].index(n)
            print(f"\n{name}:")
            print(f"Time: {results[name]['time'][idx]:.4f} ± {results[name]['time_std'][idx]:.4f} seconds")
            print(f"Memory: {results[name]['memory'][idx]:.2f} MB")