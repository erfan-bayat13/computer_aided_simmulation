import numpy as np
import networkx as nx
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def generate_geometric_graph(n, r):
    """
    Generate a random geometric graph in [0,1]^2 with radius r using an efficient grid-based approach
    """
    # Generate random positions
    positions = np.random.uniform(0, 1, size=(n, 2))
    
    # Create grid for spatial indexing
    cell_size = r
    grid = defaultdict(list)
    
    # Assign points to grid cells
    for idx, (x, y) in enumerate(positions):
        cell_x = int(x / cell_size)
        cell_y = int(y / cell_size)
        grid[(cell_x, cell_y)].append(idx)
    
    # Create graph
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    # Find edges efficiently using grid
    for idx1 in range(n):
        x1, y1 = positions[idx1]
        cell_x, cell_y = int(x1 / r), int(y1 / r)
        
        # Check neighboring cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell_x + dx, cell_y + dy)
                for idx2 in grid[neighbor_cell]:
                    if idx2 <= idx1:  # Avoid duplicates and self-loops
                        continue
                    x2, y2 = positions[idx2]
                    if np.sqrt((x1 - x2)**2 + (y1 - y2)**2) < r:
                        G.add_edge(idx1, idx2)
    
    return G

def analyze_graph_components(n, r, num_iterations=10):
    """
    Analyze component sizes for given parameters
    """
    results = []
    for _ in tqdm(range(num_iterations), desc=f"n={n}, r={r:.6f}"):
        G = generate_geometric_graph(n, r)
        components = sorted([len(c) for c in nx.connected_components(G)], reverse=True)
        largest = components[0] if components else 0
        second_largest = components[1] if len(components) > 1 else 0
        
        results.append({
            'n': n,
            'r': r,
            'a': r * r * np.pi * n / np.log(n),  # Reverse calculate a
            'largest_component': largest / n,  # Normalized size
            'second_largest_component': second_largest / n  # Normalized size
        })
    return results

# Parameters
n_values = [1000, 10000, 100000]  # Add 1000000 if computational resources allow
a_values = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
iterations = 10

# Run analysis
results = []
for n in n_values:
    for a in a_values:
        # Calculate r based on the formula r = sqrt(a*log(n)/(pi*n))
        r = np.sqrt(a * np.log(n) / (np.pi * n))
        results.extend(analyze_graph_components(n, r, iterations))

# Convert to DataFrame
df = pd.DataFrame(results)

# Create visualization
plt.figure(figsize=(15, 6))

# Plot for largest component
plt.subplot(1, 2, 1)
for n in n_values:
    data = df[df['n'] == n]
    mean_sizes = data.groupby('a')['largest_component'].mean()
    plt.plot(mean_sizes.index, mean_sizes.values, 'o-', label=f'n={n}')

plt.xlabel('a')
plt.ylabel('Normalized size of largest component')
plt.title('Largest Component Size')
plt.legend()
plt.grid(True)

# Plot for second largest component
plt.subplot(1, 2, 2)
for n in n_values:
    data = df[df['n'] == n]
    mean_sizes = data.groupby('a')['second_largest_component'].mean()
    plt.plot(mean_sizes.index, mean_sizes.values, 'o-', label=f'n={n}')

plt.xlabel('a')
plt.ylabel('Normalized size of second largest component')
plt.title('Second Largest Component Size')
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save results to CSV
df.to_csv('geometric_graph_results.csv', index=False)

if __name__ == "__main__":
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for n in n_values:
        print(f"\nFor n = {n}:")
        n_data = df[df['n'] == n]
        for a in a_values:
            a_data = n_data[n_data['a'].round(3) == a]
            print(f"\na = {a}:")
            print(f"Average largest component size: {a_data['largest_component'].mean():.3f}")
            print(f"Average second largest component size: {a_data['second_largest_component'].mean():.3f}")