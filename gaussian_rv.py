import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

class GaussianGenerator:
    def __init__(self, method='box_muller'):
        """
        Initialize the GaussianGenerator class with the chosen method for generating Gaussian random variables.
        
        Args:
            method (str): The method to use for generating Gaussian RVs. Options are:
                          'box_muller', 'clt', 'newton'.
        """
        self.method = method
    
    def generate(self, num_samples=10000):
        """
        Generate a specified number of Gaussian random variables using the chosen method.
        
        Args:
            num_samples (int): Number of Gaussian random variables to generate.
            
        Returns:
            list: Generated samples.
        """
        if self.method == 'box_muller':
            return [self.box_muller() for _ in range(num_samples // 2) for _ in range(2)]  # Generates two samples per iteration
        elif self.method == 'clt':
            return [self.central_limit_theorem() for _ in range(num_samples)]
        elif self.method == 'newton':
            return [self.newton_method() for _ in range(num_samples)]
        else:
            raise ValueError("Unknown method. Choose 'box_muller', 'clt', or 'newton'.")
    
    def box_muller(self):
        """Generate two independent standard normal random variables using the Box-Muller transformation."""
        U1 = np.random.uniform(0, 1)
        U2 = np.random.uniform(0, 1)
        Z1 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
        Z2 = np.sqrt(-2 * np.log(U1)) * np.sin(2 * np.pi * U2)
        return Z1, Z2
    
    def central_limit_theorem(self, N=12):
        """Generate a Gaussian random variable N(0,1) using the Central Limit Theorem (CLT)."""
        U = np.random.uniform(0, 1, N)
        Z = (np.sum(U) - N / 2) / np.sqrt(N / 12)
        return Z
    
    def newton_method(self, tol=1e-6, max_iter=100):
        """
        Generate a Gaussian random variable using Newton's method to invert the CDF.
        
        Args:
            tol (float): Tolerance for convergence.
            max_iter (int): Maximum number of iterations.
        
        Returns:
            float: Generated Gaussian random variable.
        """
        U = np.random.uniform(0, 1)  # Uniform random variable
        x0 = 0  # Initial guess for the root (starting at the mean of N(0,1))
        
        for _ in range(max_iter):
            fx = 0.5 * (1 + sp.erf(x0 / np.sqrt(2))) - U  # CDF of standard normal minus U
            f_prime_x = np.exp(-x0**2 / 2) / np.sqrt(2 * np.pi)  # Derivative of CDF (PDF of N(0,1))
            
            x1 = x0 - fx / f_prime_x  # Newton's method update
            
            if abs(x1 - x0) < tol:
                return x1
            x0 = x1
        
        return x0  # If no convergence, return last approximation
    
    def plot_samples(self, samples, bins=50, show_pdf=True):
        """
        Plot a histogram of the generated samples.
        
        Args:
            samples (list): The generated Gaussian samples to plot.
            bins (int): Number of bins for the histogram.
            show_pdf (bool): Whether to plot the theoretical PDF of the standard normal distribution for comparison.
        """
        plt.figure(figsize=(8, 6))
        plt.hist(samples, bins=bins, density=True, alpha=0.6, color='g', label='Generated Samples')
        
        if show_pdf:
            x_vals = np.linspace(min(samples), max(samples), 1000)
            pdf_vals = (1 / np.sqrt(2 * np.pi)) * np.exp(-x_vals**2 / 2)
            plt.plot(x_vals, pdf_vals, 'b-', lw=2, label='Theoretical N(0,1) PDF')
        
        plt.title('Generated Gaussian Samples')
        plt.xlabel('x')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def test_distribution(self, samples):
        """Test how the samples are distributed using a Q-Q plot and K-S test."""
        plt.figure(figsize=(8, 6))
        stats.probplot(samples, dist="norm", plot=plt)
        plt.title('Q-Q Plot')
        plt.grid(True)
        plt.show()
        
        ks_stat, p_value = stats.kstest(samples, 'norm')
        print(f"K-S Test Statistic: {ks_stat:.4f}, p-value: {p_value:.4f}")
        if p_value > 0.05:
            print("The samples pass the K-S test (p > 0.05), indicating they follow a normal distribution.")
        else:
            print("The samples fail the K-S test (p <= 0.05), indicating they may not follow a normal distribution.")
    
    def chi_square_test(self, samples, bins=10):
        """
        Perform a chi-square test to check how well the generated samples fit a standard normal distribution.
        
        Args:
            samples (list): The generated Gaussian samples.
            bins (int): Number of bins to use for the chi-square test.
        
        Returns:
            float: Chi-square statistic.
            float: p-value of the test.
        """
        # Compute observed frequencies by binning the data
        observed_freq, bin_edges = np.histogram(samples, bins=bins, density=False)
        
        # Compute expected frequencies using the standard normal distribution
        total_samples = len(samples)
        expected_freq = []
        for i in range(bins):
            # Calculate the cumulative probabilities for the bin edges
            cdf_min = stats.norm.cdf(bin_edges[i])
            cdf_max = stats.norm.cdf(bin_edges[i+1])
            
            # Expected count in this bin is the difference in CDF times the total number of samples
            expected_freq.append((cdf_max - cdf_min) * total_samples)
        
        # Normalize expected frequencies to match the total number of observed samples
        expected_freq = np.array(expected_freq)
        expected_freq *= total_samples / expected_freq.sum()  # Ensure sums match exactly
        
        # Perform the chi-square test
        chi_square_stat, p_value = stats.chisquare(f_obs=observed_freq, f_exp=expected_freq)
        
        # Output the results
        print(f"Chi-Square Statistic: {chi_square_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        
        if p_value > 0.05:
            print("The samples pass the chi-square test (p > 0.05), indicating they fit the normal distribution.")
        else:
            print("The samples fail the chi-square test (p <= 0.05), indicating they may not fit the normal distribution.")
        
        return chi_square_stat, p_value
    
    def test_independence(self, samples):
        """Test the independence of samples using autocorrelation and a runs test."""
        lag = 1
        autocorr = np.corrcoef(samples[:-lag], samples[lag:])[0, 1]
        print(f"Autocorrelation at lag {lag}: {autocorr:.4f}")
        if np.abs(autocorr) < 0.05:
            print("The autocorrelation is close to zero, indicating the samples are likely independent.")
        else:
            print("The autocorrelation is significantly different from zero, indicating potential dependence.")
        
        runs_stat, p_value = stats.mstats.normaltest(samples)
        print(f"Runs Test p-value: {p_value:.4f}")
        if p_value > 0.05:
            print("The samples pass the runs test (p > 0.05), indicating they are likely random.")
        else:
            print("The samples fail the runs test (p <= 0.05), indicating they may not be random.")

# Example Usage
# Create an instance of the GaussianGenerator class for the Box-Muller method
gen = GaussianGenerator(method='box_muller')

# Generate 10000 samples
samples = gen.generate(num_samples=10000)

# Plot the generated samples
print("Plotting Samples...")
gen.plot_samples(samples)

# Test the distribution
print("Testing Distribution...")
gen.test_distribution(samples)

# Test the independence
print("\nTesting Independence...")
gen.test_independence(samples)