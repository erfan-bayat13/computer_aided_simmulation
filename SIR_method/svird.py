# this the svid model 
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import odeint



def svird(sate, t, params):
    pass

def vaccination_allocation(S, params):
    pass

def calculate_foi():
    # Force of infection calculations
    pass

def spatial_diffusion(X, connectivity_matrix, diffusion_coeff):
    # handle spatial diffusion between regions 
    pass


class SVIRDModel:
    def __init__(self):
        self.n_regions = 4
        self.n_risk_groups = 4

        self.params = {
            'beta': np.zeros(self.n_risk_groups),  # Transmission rates for each risk group
            'gamma': np.zeros(self.n_risk_groups),  # Recovery rates for each risk group
            'mu': np.zeros(self.n_risk_groups),    # Mortality rates for each risk group
            'nu': 0.0,    # Total vaccination rate
            'eta': 0.0,   # Vaccine efficacy
            'omega': 0.0, # Waning immunity rate
            'alpha': np.zeros(self.n_risk_groups),  # Vaccination priority coefficients
            # Diffusion coefficients
            'diffusion': {
                'S': 0.0,
                'V': 0.0,
                'I': 0.0,
                'R': 0.0
            },
            # Connectivity matrix between regions (4x4)
            'connectivity': np.zeros((self.n_regions, self.n_regions))
        }
    
    def get_index(self,compartment, region, risk_group):
        """
        Convert (compartment, region, risk_group) to index in state vector
        
        Parameters:
        compartment: str, one of 'S', 'V', 'I', 'R', 'D'
        region: int, region index (0-3)
        risk_group: int, risk group index (0-3)
        """
        compartment_idx = {'S': 0, 'V': 1, 'I': 2, 'R': 3, 'D': 4}
        base_idx = compartment_idx[compartment] * (self.n_regions * self.n_risk_groups)
        return base_idx + region * self.n_risk_groups + risk_group
    
    def get_compartment(self, state, compartment):
        """
        Extract a specific compartment from the state vector as a (n_regions, n_risk_groups) array
        """
        start_idx = self.get_index(compartment, 0, 0)
        end_idx = start_idx + (self.n_regions * self.n_risk_groups)
        return state[start_idx:end_idx].reshape(self.n_regions, self.n_risk_groups)
    
    def set_parameters(self, **kwargs):
        """
        Set model parameters
        """
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
            else:
                raise ValueError(f"Unknown parameter: {key}")
            
    def calculate_vaccination_rates(self, state):
        """
        Calculate vaccination rates for each region and risk group
        
        Parameters:
        state: current state vector
        
        Returns:
        v_rates: numpy array of shape (n_regions, n_risk_groups) containing vaccination rates
        """

        S = self.get_compartment(state, 'S') # Shape: (n_regions, n_risk_groups)

        # Calculate denominator (sum of α_r * S_{i,r} over all regions and risk groups)
        weighted_S = S * self.params['alpha']
        total_weighted_S = np.sum(weighted_S)

        if total_weighted_S == 0:
            # If no susceptible individuals, return zero vaccination rates
            return np.zeros((self.n_regions, self.n_risk_groups))
        
        v_rates = np.zeros((self.n_regions, self.n_risk_groups))
        for i in range(self.n_regions):
            for r in range(self.n_risk_groups):
                # Apply formula: v_{i,r}(t) = ν * (α_r * S_{i,r}) / (sum of α_k * S_{j,k})
                v_rates[i, r] = (self.params['nu'] * 
                                self.params['alpha'][r] * 
                                S[i, r] / total_weighted_S)
                
                # Ensure vaccination rate doesn't exceed available susceptible population
                v_rates[i, r] = min(v_rates[i, r], S[i, r])
        
        total_rate = np.sum(v_rates)
        if total_rate > 0:
            v_rates *= min(1.0, self.params['nu'] / total_rate)
        
        return v_rates
    
    def verify_vaccination_rates(self, v_rates, state):
        """
        Verify that vaccination rates satisfy all constraints
        """
        S = self.get_compartment(state, 'S')

        if np.any(v_rates < 0):
            raise ValueError("Negative vaccination rates detected")
        
        if np.any(v_rates > S):
            raise ValueError("Vaccination rates exceed available susceptible population")
        
        total_rate = np.sum(v_rates)
        if not np.isclose(total_rate, min(self.params['nu'], np.sum(S)), rtol=1e-10):
            raise ValueError("Total vaccination rate constraint violated")


    
    def initialize_state(self, initial_conditions):
        """
        Initialize the state vector based on provided initial conditions
        """
        # Total state vector length = 5 compartments * 4 regions * 4 risk groups
        state = np.zeros(5 * self.n_regions * self.n_risk_groups)
        
        for comp in ['S', 'V', 'I', 'R', 'D']:
            if comp in initial_conditions:
                for i in range(self.n_regions):
                    for r in range(self.n_risk_groups):
                        idx = self.get_index(comp, i, r)
                        state[idx] = initial_conditions[comp][i, r]
        
        return state
    
    def verify_state(self, state):
        """
        Verify that the state vector satisfies basic constraints:
        - Non-negative values
        - Conservation of population
        """
        if np.any(state < 0):
            raise ValueError("Negative values in state vector")
        
        # Check conservation for each region and risk group
        for i in range(self.n_regions):
            for r in range(self.n_risk_groups):
                total = sum(state[self.get_index(comp, i, r)] 
                          for comp in ['S', 'V', 'I', 'R', 'D'])
                if not np.isclose(total, self.initial_total_population[i, r]):
                    raise ValueError(f"Population not conserved for region {i}, risk group {r}")
                
    def calculate_foi(self, state):
        """
        Calculate force of infection for each region and risk group
        
        The force of infection accounts for:
        - Risk group specific transmission rates (beta_r)
        - Contact between different risk groups
        - Reduced transmission to vaccinated individuals (1-eta)
        
        Returns:
        foi: numpy array of shape (n_regions, n_risk_groups) with force of infection values
        """

        # Extract needed compartments
        S = self.get_compartment(state, 'S')  # Shape: (n_regions, n_risk_groups)
        I = self.get_compartment(state, 'I')
        V = self.get_compartment(state, 'V')

        N  = np.sum(S+I+V , axis=1)
        N = N.reshape(-1,1)

        foi = np.zeros((self.n_regions,self.n_risk_groups))

        for i in range(self.n_regions):
            if N[i] == 0:
                continue
            
            # Calculate total infectious pressure in region i
            infectious_pressure = np.sum(
                self.params['beta'] * I[i] / N[i]
            )
            
            for r in range(self.n_risk_groups):
                foi[i,r] = self.params['beta'][r] * infectious_pressure
        
        return foi
    
    def verify_foi(self, foi, state):
        if np.any(foi < 0):
            raise ValueError("Negative force of infection detected") 
        
        if not np.all(np.isfinite(foi)):
            raise ValueError("Non-finite force of infection values detected")
        
        # TODO: additional might be added later

        return True
    
    def calculate_spatial_diffusion(self, state):
        """
        Calculate spatial diffusion terms for all compartments
        
        Parameters:
        state: current state vector
        
        Returns:
        diffusion_terms: dictionary with diffusion terms for each compartment
        """
        diffusion_terms = {
            'S': np.zeros((self.n_regions, self.n_risk_groups)),
            'V': np.zeros((self.n_regions, self.n_risk_groups)),
            'I': np.zeros((self.n_regions, self.n_risk_groups)),
            'R': np.zeros((self.n_regions, self.n_risk_groups))
        }

        for comp in ['S','V','I','R']:
            X = self.get_compartment(state, comp)
            d = self.params['diffusion'][comp]

            # TODO: make this code more efficent its just too bad right now
            for i in range(self.n_regions):
                for r in range(self.n_risk_groups):
                    # Sum up contributions from all connected regions
                    diffusion = 0
                    for j in range(self.n_regions):
                        if i != j:  # Skip self-connection
                            # Flow from j to i minus flow from i to j
                            diffusion += d * self.params['connectivity'][i,j] * (X[j,r] - X[i,r])
                    
                    diffusion_terms[comp][i,r] = diffusion
        return diffusion_terms
    
    def calculate_derivatives(self, state, t):
        """
        Calculate the derivatives for the SVIRD system
        
        Parameters:
        state: current state vector
        t: current time
        
        Returns:
        derivatives: state vector of same shape as input with computed derivatives
        """
        # Initialize derivative vector
        derivatives = np.zeros_like(state)
        
        # Get current values for each compartment
        S = self.get_compartment(state, 'S')
        V = self.get_compartment(state, 'V')
        I = self.get_compartment(state, 'I')
        R = self.get_compartment(state, 'R')
        
        # Calculate intermediate terms
        foi = self.calculate_foi(state)
        v_rates = self.calculate_vaccination_rates(state)
        diffusion = self.calculate_spatial_diffusion(state)

        for i in range(self.n_regions):
            for r in range(self.n_risk_groups):
                s_idx = self.get_index('S', i, r)
                v_idx = self.get_index('V', i, r)
                i_idx = self.get_index('I', i, r)
                r_idx = self.get_index('R', i, r)
                d_idx = self.get_index('D', i, r)

                derivatives[s_idx] = (
                    -foi[i,r] * S[i,r]
                    - v_rates[i,r]
                    + self.params['omega'] * V[i,r]
                    + diffusion['S'][i,r]
                )

                # Vaccinated compartment
                derivatives[v_idx] = (
                    v_rates[i,r]                                    # Vaccination
                    - (1 - self.params['eta']) * foi[i,r] * V[i,r] # Breakthrough infections
                    - self.params['omega'] * V[i,r]                 # Waning immunity
                    + diffusion['V'][i,r]                          # Spatial diffusion
                )


                # Infected compartment
                derivatives[i_idx] = (
                    foi[i,r] * S[i,r]                              # Infection of susceptible
                    + (1 - self.params['eta']) * foi[i,r] * V[i,r] # Breakthrough infections
                    - self.params['gamma'][r] * I[i,r]             # Recovery
                    - self.params['mu'][r] * I[i,r]                # Deaths
                    + diffusion['I'][i,r]                          # Spatial diffusion
                )
                
                # Recovered compartment
                derivatives[r_idx] = (
                    self.params['gamma'][r] * I[i,r]  # Recovery
                    + diffusion['R'][i,r]             # Spatial diffusion
                )
                
                # Deaths compartment
                derivatives[d_idx] = (
                    self.params['mu'][r] * I[i,r]     # Deaths
                )
        
        return derivatives
    
    def solve(self, initial_conditions, t_span, t_eval=None):
        """
        Solve the SVIRD system over the specified time period
        
        Parameters:
        initial_conditions: dictionary with initial values for each compartment
        t_span: (t_start, t_end) tuple
        t_eval: optional array of evaluation times
        
        Returns:
        t: time points
        solution: array with solution values
        """
        y0 = self.initialize_state(initial_conditions=initial_conditions)

        solution = odeint(self.calculate_derivatives,
                          y0,
                          t_eval if t_eval is not None else np.linspace(*t_span, 100),
                          rtol=1e-8,
                          atol=1e-8
                        )
        return t_eval if t_eval is not None else np.linspace(*t_span, 100), solution
    

    def test_foi_calculation(self):
        """
        Test the force of infection calculation with a simple scenario
        """
        # Set up test parameters
        self.params.update({
            'beta': np.array([0.3, 0.4, 0.5, 0.6]),  # Increasing transmission rates
            'eta': 0.9  # 90% vaccine efficacy
        })
        
        # Create test initial conditions
        initial_conditions = {
            'S': np.ones((4, 4)) * 1000,  # 1000 susceptible in each group
            'I': np.zeros((4, 4)),
            'V': np.zeros((4, 4)),
            'R': np.zeros((4, 4)),
            'D': np.zeros((4, 4))
        }
        
        # Add some infected individuals
        initial_conditions['I'][0] = [10, 20, 30, 40]
        
        # Region 1: Even distribution
        initial_conditions['I'][1] = [25, 25, 25, 25]
        
        # Region 2: More infections in low-risk groups
        initial_conditions['I'][2] = [40, 30, 20, 10]
        
        # Region 3: Mixed pattern
        initial_conditions['I'][3] = [15, 35, 15, 35]
        
        # Initialize state
        state = self.initialize_state(initial_conditions)
        
        # Calculate force of infection
        foi = self.calculate_foi(state)
        
        print("\nForce of Infection Test Results:")
        print("-" * 50)
        print("\nTransmission rates (β):", self.params['beta'])
        print("\nForce of infection by region and risk group:")
        print(foi)
        
        # Verify results
        self.verify_foi(foi, state)
        print("\nAll force of infection calculations are valid!")
        
        return foi

    def plot_foi(self, foi):
        """
        Visualize force of infection across regions and risk groups
        """
        plt.figure(figsize=(10, 6))
        
        # Create heatmap
        plt.imshow(foi, cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='Force of Infection')
        
        # Add labels
        plt.xlabel('Risk Group')
        plt.ylabel('Region')
        plt.title('Force of Infection Heatmap')
        
        # Add tick labels
        plt.xticks(range(self.n_risk_groups), [f'Group {i}' for i in range(self.n_risk_groups)])
        plt.yticks(range(self.n_regions), [f'Region {i}' for i in range(self.n_regions)])
        
        plt.show()

    def test_spatial_diffusion(self):
        """
        Test spatial diffusion calculations with a simple scenario
        """
        # Set up test parameters
        self.params.update({
            'diffusion': {
                'S': 0.1,  # Diffusion coefficient for susceptible
                'V': 0.05, # Lower diffusion for vaccinated
                'I': 0.01, # Very low diffusion for infected
                'R': 0.1   # Same as susceptible
            },
            # Example connectivity matrix (symmetric)
            'connectivity': np.array([
                [0, 1, 0.5, 0],    # Region 0 connected to 1 and 2
                [1, 0, 1, 0.5],    # Region 1 connected to 0, 2, and 3
                [0.5, 1, 0, 1],    # Region 2 connected to 0, 1, and 3
                [0, 0.5, 1, 0]     # Region 3 connected to 1 and 2
            ])
        })
        
        # Create test initial conditions with population gradient
        initial_conditions = {
            'S': np.array([
                [1000, 800, 600, 400],    # Region 0 - highest population
                [800, 600, 400, 200],     # Region 1
                [600, 400, 200, 100],     # Region 2
                [400, 200, 100, 50]       # Region 3 - lowest population
            ]),
            'I': np.zeros((4, 4)),
            'V': np.zeros((4, 4)),
            'R': np.zeros((4, 4)),
            'D': np.zeros((4, 4))
        }
        
        # Initialize state
        state = self.initialize_state(initial_conditions)
        
        # Calculate diffusion terms
        diffusion_terms = self.calculate_spatial_diffusion(state)
        
        # Print results
        print("\nSpatial Diffusion Test Results:")
        print("-" * 50)
        print("\nDiffusion coefficients:")
        for comp, d in self.params['diffusion'].items():
            print(f"{comp}: {d}")
        
        print("\nConnectivity matrix:")
        print(self.params['connectivity'])
        
        print("\nInitial susceptible population:")
        print(self.get_compartment(state, 'S'))
        
        print("\nDiffusion terms for susceptible population:")
        print(diffusion_terms['S'])
        
        # Verify conservation
        for comp in ['S', 'V', 'I', 'R']:
            total_flow = np.sum(diffusion_terms[comp])
            print(f"\nTotal flow for {comp}: {total_flow:.2e}")
            assert np.abs(total_flow) < 1e-10, f"Conservation violated for {comp}"
        
        return diffusion_terms
    
    def test_simulation(self, t_max=100):
        """
        Run a test simulation with reasonable parameters
        """
        # Set up parameters
        self.params.update({
            'beta': np.array([0.3, 0.4, 0.5, 0.6]),  # Transmission rates
            'gamma': np.array([0.1, 0.1, 0.1, 0.1]), # Recovery rates
            'mu': np.array([0.01, 0.02, 0.03, 0.04]), # Mortality rates
            'nu': 50.0,  # Vaccination capacity
            'eta': 0.9,  # Vaccine efficacy
            'omega': 0.01, # Waning immunity rate
            'alpha': np.array([1.0, 2.0, 3.0, 4.0]),  # Vaccination priorities
            'diffusion': {
                'S': 0.1,
                'V': 0.05,
                'I': 0.01,
                'R': 0.1
            },
            'connectivity': np.array([
                [0, 1, 0.5, 0],
                [1, 0, 1, 0.5],
                [0.5, 1, 0, 1],
                [0, 0.5, 1, 0]
            ])
        })
        
        # Initial conditions
        initial_conditions = {
            'S': np.ones((4, 4)) * 1000,  # 1000 susceptible in each group
            'I': np.zeros((4, 4)),
            'V': np.zeros((4, 4)),
            'R': np.zeros((4, 4)),
            'D': np.zeros((4, 4))
        }
        
        # Add initial infections
        initial_conditions['I'][0,0] = 100  # Start with 100 infected in region 0, risk group 0
        
        # Solve system
        t = np.linspace(0, t_max, 1000)
        t, solution = self.solve(initial_conditions, (0, t_max), t)
        
        return t, solution

    def plot_diffusion_terms(self, diffusion_terms):
        """
        Visualize spatial diffusion terms for all compartments
        """
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        compartments = ['S', 'V', 'I', 'R']
        
        for idx, (comp, ax) in enumerate(zip(compartments, axes.flat)):
            im = ax.imshow(diffusion_terms[comp], cmap='RdBu', aspect='auto')
            plt.colorbar(im, ax=ax, label='Net Flow')
            
            ax.set_title(f'Diffusion Terms - {comp} Compartment')
            ax.set_xlabel('Risk Group')
            ax.set_ylabel('Region')
            ax.set_xticks(range(self.n_risk_groups))
            ax.set_yticks(range(self.n_regions))
            ax.set_xticklabels([f'Group {i}' for i in range(self.n_risk_groups)])
            ax.set_yticklabels([f'Region {i}' for i in range(self.n_regions)])
            
        plt.tight_layout()
        plt.show()

    def plot_simulation(self, t, solution):
        """
        Plot the results of a simulation
        """
        compartments = ['S', 'V', 'I', 'R', 'D']
        fig, axes = plt.subplots(len(compartments), 1, figsize=(12, 15))
        
        for idx, comp in enumerate(compartments):
            ax = axes[idx]
            
            for i in range(self.n_regions):
                for r in range(self.n_risk_groups):
                    state_idx = self.get_index(comp, i, r)
                    ax.plot(t, solution[:, state_idx], 
                           label=f'Region {i}, Risk {r}',
                           alpha=0.7)
            
            ax.set_title(f'{comp} Compartment')
            ax.set_xlabel('Time')
            ax.set_ylabel('Population')
            ax.grid(True)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
    
    def plot_region_comparison(self, t, solution):
        """
        Plot comparison of total cases across regions
        """
        plt.figure(figsize=(12, 8))
        
        for i in range(self.n_regions):
            # Sum up infected across risk groups for each region
            total_infected = sum(
                solution[:, self.get_index('I', i, r)]
                for r in range(self.n_risk_groups)
            )
            plt.plot(t, total_infected, label=f'Region {i}', linewidth=2)
        
        plt.title('Total Infected Cases by Region')
        plt.xlabel('Time')
        plt.ylabel('Number of Infected')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_risk_group_comparison(self, t, solution):
        """
        Plot comparison of cases across risk groups
        """
        plt.figure(figsize=(12, 8))
        
        for r in range(self.n_risk_groups):
            # Sum up infected across regions for each risk group
            total_infected = sum(
                solution[:, self.get_index('I', i, r)]
                for i in range(self.n_regions)
            )
            plt.plot(t, total_infected, label=f'Risk Group {r}', linewidth=2)
        
        plt.title('Total Infected Cases by Risk Group')
        plt.xlabel('Time')
        plt.ylabel('Number of Infected')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_vaccination_progress(self, t, solution):
        """
        Plot vaccination progress over time
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot total vaccinated population
        for i in range(self.n_regions):
            total_vaccinated = sum(
                solution[:, self.get_index('V', i, r)]
                for r in range(self.n_risk_groups)
            )
            ax1.plot(t, total_vaccinated, label=f'Region {i}')
        
        ax1.set_title('Vaccination Progress by Region')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Number of Vaccinated Individuals')
        ax1.legend()
        ax1.grid(True)
        
        # Plot vaccination by risk group
        for r in range(self.n_risk_groups):
            total_vaccinated = sum(
                solution[:, self.get_index('V', i, r)]
                for i in range(self.n_regions)
            )
            ax2.plot(t, total_vaccinated, label=f'Risk Group {r}')
        
        ax2.set_title('Vaccination Progress by Risk Group')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Number of Vaccinated Individuals')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_mortality_analysis(self, t, solution):
        """
        Plot mortality rates and cumulative deaths
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot deaths over time by region
        for i in range(self.n_regions):
            total_deaths = sum(
                solution[:, self.get_index('D', i, r)]
                for r in range(self.n_risk_groups)
            )
            ax1.plot(t, total_deaths, label=f'Region {i}')
        
        ax1.set_title('Cumulative Deaths by Region')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Number of Deaths')
        ax1.legend()
        ax1.grid(True)
        
        # Plot deaths by risk group
        for r in range(self.n_risk_groups):
            total_deaths = sum(
                solution[:, self.get_index('D', i, r)]
                for i in range(self.n_regions)
            )
            ax2.plot(t, total_deaths, label=f'Risk Group {r}')
        
        ax2.set_title('Cumulative Deaths by Risk Group')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Number of Deaths')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_spatial_heatmap(self, t, solution, time_points=None):
        """
        Plot heatmap of infected cases across regions at different time points
        """
        if time_points is None:
            time_points = [0, len(t)//4, len(t)//2, 3*len(t)//4, -1]
        
        fig, axes = plt.subplots(1, len(time_points), figsize=(20, 4))
        
        for idx, t_idx in enumerate(time_points):
            data = np.zeros((self.n_regions, self.n_risk_groups))
            for i in range(self.n_regions):
                for r in range(self.n_risk_groups):
                    data[i, r] = solution[t_idx, self.get_index('I', i, r)]
            
            im = axes[idx].imshow(data, cmap='YlOrRd')
            axes[idx].set_title(f'Time = {t[t_idx]:.1f}')
            axes[idx].set_xlabel('Risk Group')
            axes[idx].set_ylabel('Region')
            plt.colorbar(im, ax=axes[idx])
        
        plt.suptitle('Spatial Distribution of Infected Cases Over Time')
        plt.tight_layout()
        plt.show()

    def plot_peak_timing_analysis(self, t, solution):
        """
        Plot analysis of epidemic peak timing across regions and risk groups
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Peak timing by region
        peak_times_region = []
        peak_values_region = []
        for i in range(self.n_regions):
            total_infected = sum(solution[:, self.get_index('I', i, r)]
                               for r in range(self.n_risk_groups))
            peak_idx = np.argmax(total_infected)
            peak_times_region.append(t[peak_idx])
            peak_values_region.append(total_infected[peak_idx])
        
        ax1.bar(range(self.n_regions), peak_times_region)
        ax1.set_title('Peak Timing by Region')
        ax1.set_xlabel('Region')
        ax1.set_ylabel('Time to Peak')
        
        # Peak timing by risk group
        peak_times_risk = []
        peak_values_risk = []
        for r in range(self.n_risk_groups):
            total_infected = sum(solution[:, self.get_index('I', i, r)]
                               for i in range(self.n_regions))
            peak_idx = np.argmax(total_infected)
            peak_times_risk.append(t[peak_idx])
            peak_values_risk.append(total_infected[peak_idx])
        
        ax2.bar(range(self.n_risk_groups), peak_times_risk)
        ax2.set_title('Peak Timing by Risk Group')
        ax2.set_xlabel('Risk Group')
        ax2.set_ylabel('Time to Peak')
        
        plt.tight_layout()
        plt.show()

    def plot_all_compartments_combined(self, t, solution):
        """
        Plot all SVIRD compartments on a single figure with both total populations
        and stacked area plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Colors for each compartment
        colors = {
            'S': 'blue',
            'V': 'green',
            'I': 'red',
            'R': 'purple',
            'D': 'black'
        }
        
        # Calculate totals for each compartment
        totals = {}
        for comp in ['S', 'V', 'I', 'R', 'D']:
            totals[comp] = np.sum([
                [solution[:, self.get_index(comp, i, r)]
                 for r in range(self.n_risk_groups)]
                for i in range(self.n_regions)
            ], axis=(0, 1))
        
        # Line plot
        for comp, color in colors.items():
            ax1.plot(t, totals[comp], label=comp, color=color, linewidth=2)
        
        ax1.set_title('Total Population in Each Compartment')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Population')
        ax1.legend()
        ax1.grid(True)
        
        # Stacked area plot
        compartments = ['S', 'V', 'I', 'R', 'D']
        data = [totals[comp] for comp in compartments]
        
        ax2.stackplot(t, data, labels=compartments, colors=[colors[comp] for comp in compartments],
                     alpha=0.7)
        
        ax2.set_title('Stacked Population Distribution')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Population')
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax2.grid(True)
        
        # Add total population line to verify conservation
        total_population = sum(data)
        ax2.plot(t, total_population, '--', color='gray', label='Total Population',
                linewidth=1, alpha=0.8)
        
        plt.tight_layout()
        plt.show()
        
        # Print some summary statistics
        print("\nSummary Statistics:")
        print("-" * 50)
        for comp in compartments:
            print(f"\n{comp} Compartment:")
            print(f"  Peak: {np.max(totals[comp]):.0f}")
            print(f"  Final value: {totals[comp][-1]:.0f}")
        print("\nPopulation Conservation Check:")
        print(f"  Initial total: {total_population[0]:.0f}")
        print(f"  Final total: {total_population[-1]:.0f}")
    

# Test the implementation
if __name__ == "__main__":
    model = SVIRDModel()
    t, solution = model.test_simulation()
    model.plot_simulation(t, solution)