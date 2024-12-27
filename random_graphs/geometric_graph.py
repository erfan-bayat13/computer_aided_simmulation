import numpy as np
import networkx as nx
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

def generate_geometric_graph(n, r):
    """
    Generate a random geometric graph in [0,1]^2 with radius r
    """
    # Generate random positions
    positions = np.random.uniform(0, 1, size=(n, 2))
    
    # Calculate pairwise distances
    distances = squareform(pdist(positions))
    
    # Create adjacency matrix based on distance threshold
    adjacency = (distances < r) & (distances > 0)
    
    # Create NetworkX graph
    G = nx.from_numpy_array(adjacency)
    
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
n_values = [1000, 10000]  # Add 1000000 if computational resources allow
a_values = [0.8, 0.9, 1.0, 1.1, 1.2]
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