# this the svid model 
from matplotlib import pyplot as plt
import numpy as np


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

# Test the implementation
if __name__ == "__main__":
    model = SVIRDModel()
    foi = model.test_foi_calculation()
    model.plot_foi(foi)