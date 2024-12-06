import numpy as np
from scipy.stats import poisson, binom, geom, norm
import matplotlib.pyplot as plt

def galton_watson_simulation(Y_v, m, N=10000, max_generations=50):
    """
    Simulate the Galton-Watson process for extinction probabilities and population sizes.
    """
    def sample_offspring(Y_v, m):
        if Y_v == "poisson":
            return np.random.poisson(m)
        elif Y_v == "binomial":
            return np.random.binomial(10, m / 10)
        elif Y_v == "geometric":
            return np.random.geometric(1 / m) - 1  # Zero-indexed
        else:
            raise ValueError("Unsupported offspring distribution")
    
    extinction_counts = np.zeros(max_generations)
    population_sizes = np.zeros((N, max_generations))
    
    for sim in range(N):
        Z = 1  # Start with one ancestor
        for gen in range(max_generations):
            population_sizes[sim, gen] = Z
            if Z == 0:
                extinction_counts[gen:] += 1
                break
            Z = sum(sample_offspring(Y_v, m) for _ in range(int(Z)))
    
    # Estimate q_i and q
    qi = extinction_counts / N
    q = qi[-1]
    
    # Compute confidence intervals
    z = norm.ppf(0.975)  # 95% CI
    ci = [
        (q_gen - z * np.sqrt(q_gen * (1 - q_gen) / N),
         q_gen + z * np.sqrt(q_gen * (1 - q_gen) / N)) 
        for q_gen in qi
    ]
    
    return qi, q, ci, population_sizes

def plot_extinction(qi, ci, Y_v, m):
    """
    Plot extinction probabilities and confidence intervals.
    """
    generations = np.arange(len(qi))
    lower_ci = [c[0] for c in ci]
    upper_ci = [c[1] for c in ci]
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, qi, label="Extinction Probability ($q_i$)", marker='o')
    plt.fill_between(generations, lower_ci, upper_ci, color='gray', alpha=0.3, label="95% CI")
    plt.axhline(qi[-1], color='red', linestyle='--', label=f"Overall Extinction ($q = {qi[-1]:.3f}$)")
    plt.title(f"Extinction Probability ($q_i$) - {Y_v.capitalize()} Distribution (m={m})")
    plt.xlabel("Generation")
    plt.ylabel("Extinction Probability ($q_i$)")
    plt.legend()
    plt.grid()
    plt.show()

def plot_population_size(population_sizes, Y_v, m):
    """
    Plot the average population size and standard deviation over generations.
    """
    mean_population = np.mean(population_sizes, axis=0)
    std_population = np.std(population_sizes, axis=0)
    generations = np.arange(len(mean_population))
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, mean_population, label="Mean Population Size", color="blue", marker='o')
    plt.fill_between(
        generations, 
        mean_population - std_population, 
        mean_population + std_population, 
        color="blue", 
        alpha=0.2, 
        label="1 Std Dev"
    )
    plt.title(f"Average Population Size - {Y_v.capitalize()} Distribution (m={m})")
    plt.xlabel("Generation")
    plt.ylabel("Population Size")
    plt.legend()
    plt.grid()
    plt.show()

# Example usage with both plots
Y_v_list = ["poisson", "binomial", "geometric"]
m_values = [1.05, 1.1]

for Y_v in Y_v_list:
    for m in m_values:
        print(f"Simulating: Distribution={Y_v}, m={m}")
        qi, q, ci, population_sizes = galton_watson_simulation(Y_v, m, N=10000, max_generations=50)
        print(f"q (Overall Extinction Probability): {q}")
        print(f"CI for q: {ci[-1]}")
        
        # Extinction Probability Plot
        plot_extinction(qi, ci, Y_v, m)
        
        # Population Size Plot
        plot_population_size(population_sizes, Y_v, m)
