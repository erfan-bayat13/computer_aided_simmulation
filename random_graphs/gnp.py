import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

def analyze_graph_components(n, p, num_iterations=10):
    results = []
    for _ in range(num_iterations):
        G = nx.fast_gnp_random_graph(n, p)
        components = sorted([len(c) for c in nx.connected_components(G)], reverse=True)
        largest = components[0] if components else 0
        second_largest = components[1] if len(components) > 1 else 0
        results.append({
            'n': n,
            'p': p,
            'largest_component': largest / n,  # Normalized size
            'second_largest_component': second_largest / n  # Normalized size
        })
    return results

# Parameters
n_values = [1000, 10000, 100000]  # 10^6 is too computationally intensive
a_values = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
iterations = 10

# Part 1: p = a/n
results_linear = []
for n in n_values:
    for a in a_values:
        p = a/n
        results_linear.extend(analyze_graph_components(n, p, iterations))

# Part 2: p = a*log(n)/n
results_log = []
for n in n_values:
    for a in a_values:
        p = a * np.log(n)/n
        # Store the original 'a' value for plotting
        results_extend = analyze_graph_components(n, p, iterations)
        for r in results_extend:
            r['a'] = a  # Store the original 'a' value
        results_log.extend(results_extend)

# Convert to DataFrames
df_linear = pd.DataFrame(results_linear)
df_log = pd.DataFrame(results_log)

# Plotting function for linear case
def create_linear_component_plots(df, title_prefix):
    plt.figure(figsize=(15, 6))
    
    # Plot for largest component
    plt.subplot(1, 2, 1)
    for n in n_values:
        data = df[df['n'] == n]
        mean_sizes = data.groupby('p')['largest_component'].mean()
        plt.plot(mean_sizes.index * n, mean_sizes.values, 'o-', label=f'n={n}')
    
    plt.xlabel('a')
    plt.ylabel('Normalized size of largest component')
    plt.title(f'{title_prefix}\nLargest Component Size')
    plt.legend()
    plt.grid(True)
    
    # Plot for second largest component
    plt.subplot(1, 2, 2)
    for n in n_values:
        data = df[df['n'] == n]
        mean_sizes = data.groupby('p')['second_largest_component'].mean()
        plt.plot(mean_sizes.index * n, mean_sizes.values, 'o-', label=f'n={n}')
    
    plt.xlabel('a')
    plt.ylabel('Normalized size of second largest component')
    plt.title(f'{title_prefix}\nSecond Largest Component Size')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    return plt

# Modified plotting function for logarithmic case
def create_log_component_plots(df, title_prefix):
    plt.figure(figsize=(15, 6))
    
    # Plot for largest component
    plt.subplot(1, 2, 1)
    for n in n_values:
        data = df[df['n'] == n]
        mean_sizes = data.groupby('a')['largest_component'].mean()
        plt.plot(mean_sizes.index, mean_sizes.values, 'o-', label=f'n={n}')
    
    plt.xlabel('a (where p = a*log(n)/n)')
    plt.ylabel('Normalized size of largest component')
    plt.title(f'{title_prefix}\nLargest Component Size')
    plt.legend()
    plt.grid(True)
    
    # Plot for second largest component
    plt.subplot(1, 2, 2)
    for n in n_values:
        data = df[df['n'] == n]
        mean_sizes = data.groupby('a')['second_largest_component'].mean()
        plt.plot(mean_sizes.index, mean_sizes.values, 'o-', label=f'n={n}')
    
    plt.xlabel('a (where p = a*log(n)/n)')
    plt.ylabel('Normalized size of second largest component')
    plt.title(f'{title_prefix}\nSecond Largest Component Size')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    return plt

# Create plots
fig1 = create_linear_component_plots(df_linear, 'Linear Case (p = a/n)')
fig2 = create_log_component_plots(df_log, 'Logarithmic Case (p = a*log(n)/n)')