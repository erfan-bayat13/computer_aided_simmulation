# this the svid model 
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