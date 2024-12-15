from svird import *

import numpy as np
from matplotlib import pyplot as plt

def test_vaccination_allocation():
    # Initialize model
    model = SVIRDModel()
    
    # Set up test parameters
    model.params.update({
        'nu': 100.0,  # Total vaccination capacity of 100 people per time unit
        'alpha': np.array([1.0, 2.0, 3.0, 4.0]),  # Higher priority for higher risk groups
    })
    
    # Create test initial conditions
    # We'll create a scenario with different susceptible populations
    initial_conditions = {
        'S': np.array([
            [1000, 800, 600, 400],    # Region 0
            [900, 700, 500, 300],     # Region 1
            [800, 600, 400, 200],     # Region 2
            [700, 500, 300, 100]      # Region 3
        ]),
        'V': np.zeros((4, 4)),
        'I': np.zeros((4, 4)),
        'R': np.zeros((4, 4)),
        'D': np.zeros((4, 4))
    }
    
    # Initialize state
    state = model.initialize_state(initial_conditions)
    
    # Calculate vaccination rates
    v_rates = model.calculate_vaccination_rates(state)
    
    # Print results
    print("\nVaccination Allocation Test Results:")
    print("-" * 50)
    print("\nPriority coefficients (α):", model.params['alpha'])
    print("\nSusceptible population by region and risk group:")
    print(initial_conditions['S'])
    print("\nCalculated vaccination rates:")
    print(v_rates)
    print("\nTotal vaccination rate:", np.sum(v_rates))
    
    # Verify results
    model.verify_vaccination_rates(v_rates, state)
    print("\nAll vaccination constraints satisfied!")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    # Create bar plot
    x = np.arange(4)
    width = 0.2
    
    for region in range(4):
        plt.bar(x + region * width, v_rates[region], width, 
                label=f'Region {region}',
                alpha=0.7)
    
    plt.xlabel('Risk Group')
    plt.ylabel('Vaccination Rate')
    plt.title('Vaccination Allocation by Region and Risk Group')
    plt.legend()
    plt.xticks(x + 1.5 * width, [f'Group {i}' for i in range(4)])
    plt.grid(True, alpha=0.3)
    plt.show()

# Run the test
if __name__ == "__main__":
    test_vaccination_allocation()