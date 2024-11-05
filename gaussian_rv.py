import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.special as sp
import statsmodels.api as sm
from scipy.stats import norm

class GaussianGenerator:
    def __init__(self, method='box_muller', seed=None):
        """
        Initialize the GaussianGenerator class with the chosen method for generating Gaussian random variables.
        
        Args:
            method (str): The method to use for generating Gaussian RVs. Options are:
                          'box_muller', 'clt', 'newton'.
        """
        self.method = method
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
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
    

    def autocorrelation_test(self, samples, lags=40):
        """
        Perform an autocorrelation test to check for independence of the samples.
        
        Args:
            samples (list): The generated Gaussian samples.
            lags (int): Number of lags to compute the autocorrelation for.
        
        Returns:
            None: It plots the autocorrelation function and prints the result.
        """
        # Calculate the autocorrelation
        acf = sm.tsa.acf(samples, nlags=lags, fft=True)
        
        # Plot the autocorrelation function
        plt.figure(figsize=(10, 6))
        plt.stem(range(lags+1), acf)  # Removed 'use_line_collection' argument
        plt.axhline(y=0, linestyle='--', color='gray')
        plt.title('Autocorrelation Function')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.show()
        
        # Check if autocorrelations are near zero for non-zero lags
        for lag, value in enumerate(acf[1:], start=1):
            if abs(value) > 0.1:  # Threshold for significant autocorrelation (tune as needed)
                print(f"Significant autocorrelation found at lag {lag}: {value:.4f}")
                return False
        
        print("No significant autocorrelation found. The samples appear to be independent.")
        return True
    
    def runs_test(self, samples):
        """
        Perform a runs test to check for independence of the samples.
        
        Args:
            samples (list): The generated Gaussian samples.
        
        Returns:
            float: z-statistic for the runs test.
            float: p-value for the test.
        """
        # Convert to +1 (for increasing) and -1 (for decreasing) values
        runs = np.diff(samples) > 0
        runs = runs.astype(int)
        
        # Count the number of runs
        run_changes = np.diff(runs)
        run_count = np.sum(run_changes != 0) + 1  # Number of runs
        
        # Calculate expected number of runs and standard deviation
        n1 = np.sum(runs)  # Number of increases
        n2 = len(runs) - n1  # Number of decreases
        total_runs = n1 + n2
        expected_runs = (2 * n1 * n2) / total_runs + 1
        std_runs = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - total_runs)) / (total_runs**2 * (total_runs - 1)))
        
        # Calculate z-statistic
        z_stat = (run_count - expected_runs) / std_runs
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))  # Two-tailed p-value
        
        # Output the results
        print(f"Runs Count: {run_count}")
        print(f"Expected Runs: {expected_runs:.4f}")
        print(f"z-statistic: {z_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        
        if p_value > 0.05:
            print("The runs test indicates the samples are independent (p > 0.05).")
        else:
            print("The runs test suggests the samples may not be independent (p <= 0.05).")
        
        return z_stat, p_value
    
    def test_independence(self, samples):
        """Test the independence of samples using autocorrelation and a runs test.
            in this version it uses nupmy functions instead of statsmodels, corrcoef is being used"""
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