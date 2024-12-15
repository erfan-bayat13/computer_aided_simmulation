import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def inhomogeneous_sir(state, t, beta_matrix, gamma_vec, N_vec):
    """
    Solve inhomogeneous SIR model with multiple population groups
    
    Parameters:
    state: array of [S1, S2, ..., I1, I2, ..., R1, R2, ...]
    beta_matrix: contact matrix between groups
    gamma_vec: recovery rates for each group
    N_vec: population sizes for each group
    """
    n_groups = len(N_vec)
    S = state[:n_groups]
    I = state[n_groups:2*n_groups]
    R = state[2*n_groups:]

    # Verify conservation of population
    for i in range(n_groups):
        if not np.isclose(S[i] + I[i] + R[i], N_vec[i], rtol=1e-10):
            raise ValueError(f"Population not conserved in group {i}")
    
    # Calculate force of infection for each group
    foi = np.zeros(n_groups)
    for i in range(n_groups):
        for j in range(n_groups):
            foi[i] += beta_matrix[i,j] * I[j] / N_vec[j]
    
    # Differential equations for each group
    dSdt = -S * foi
    dIdt = S * foi - gamma_vec * I
    dRdt = gamma_vec * I
    
    return np.concatenate([dSdt, dIdt, dRdt])

# Example with two groups (high-risk and low-risk)
N1, N2 = 8000, 2000  # Population sizes
beta_matrix = np.array([[0.3, 0.2],  # Higher contact within groups
                       [0.2, 0.5]])   # Group 2 has higher contact rate
gamma_vec = np.array([0.1, 0.1])     # Same recovery rate

# Initial conditions (small number infected in group 1)
I1_0, I2_0 = 100, 0
R1_0, R2_0 = 0, 0
S1_0, S2_0 = N1 - I1_0 - R1_0, N2 - I2_0 - R2_0

state0 = np.array([S1_0, S2_0, I1_0, I2_0, R1_0, R2_0])
t = np.linspace(0, 100, 1000)

# Solve system
solution = odeint(inhomogeneous_sir, state0, t, 
                 args=(beta_matrix, gamma_vec, np.array([N1, N2])))

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(t, solution[:, 0], 'b-', label='Susceptible (Group 1)')
plt.plot(t, solution[:, 1], 'b--', label='Susceptible (Group 2)')
plt.plot(t, solution[:, 2], 'r-', label='Infected (Group 1)')
plt.plot(t, solution[:, 3], 'r--', label='Infected (Group 2)')
plt.plot(t, solution[:, 4], 'g-', label='Recovered (Group 1)')
plt.plot(t, solution[:, 5], 'g--', label='Recovered (Group 2)')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Inhomogeneous SIR Model: Two Risk Groups')
plt.legend()
plt.grid(True)
plt.show()