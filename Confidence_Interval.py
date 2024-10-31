import numpy as np
from scipy import stats
from collections import defaultdict
import warnings

class ConfidenceIntervalAnalyzer:
    """
    Class to handle confidence interval calculations for simulation metrics with
    robust handling of numerical edge cases.
    """
    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level
        self.simulation_results = []
        # Suppress specific scipy warnings about invalid values
        warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    def add_simulation_run(self, metrics):
        """Add results from a single simulation run with validation."""
        try:
            # Validate and clean time in system data
            time_in_system = metrics['total_time_in_system']
            time_in_system = [t for t in time_in_system if t >= 0]  # Remove negative values
            
            run_metrics = {
                'avg_time_in_system': np.mean(time_in_system) if time_in_system else 0,
                'completion_rate': (metrics['completed_customers'] / metrics['total_customers'] 
                                 if metrics['total_customers'] > 0 else 0),
                'rejection_rate': (metrics['rejected_customers'] / metrics['total_customers']
                                if metrics['total_customers'] > 0 else 0),
                'section_metrics': {}
            }
            
            # Extract and validate section-specific metrics
            for section, section_data in metrics['section_metrics'].items():
                # Clean and validate service times
                service_times = [t for t in section_data['service_times'] if t >= 0]
                waiting_times = [t for t in section_data['waiting_times'] if t >= 0]
                utilization = [u for u in section_data['server_utilization'] if 0 <= u <= 1]
                
                run_metrics['section_metrics'][section] = {
                    'avg_service_time': np.mean(service_times) if service_times else 0,
                    'avg_waiting_time': np.mean(waiting_times) if waiting_times else 0,
                    'utilization': np.mean(utilization) if utilization else 0
                }
            
            self.simulation_results.append(run_metrics)
            
        except Exception as e:
            print(f"Error adding simulation run: {str(e)}")
    
    def _validate_data(self, data):
        """
        Validate data before calculating confidence intervals.
        
        Args:
            data (list): List of numerical values
            
        Returns:
            numpy.array: Cleaned and validated data array
        """
        if not data:
            return np.array([])
            
        # Convert to numpy array and remove any non-finite values
        data = np.array(data, dtype=float)
        data = data[np.isfinite(data)]
        
        # Remove extreme outliers (more than 3 IQR from quartiles)
        if len(data) > 4:
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            data = data[(data >= lower_bound) & (data <= upper_bound)]
        
        return data
    
    def calculate_confidence_interval(self, data):
        """
        Calculate confidence interval with robust handling of edge cases.
        
        Args:
            data (list): List of observations
            
        Returns:
            dict: Dictionary containing mean, CI bounds, and standard error
        """
        try:
            # Validate and clean data
            clean_data = self._validate_data(data)
            
            if len(clean_data) < 2:
                return {
                    'mean': np.mean(clean_data) if len(clean_data) == 1 else 0,
                    'ci_lower': 0,
                    'ci_upper': 0,
                    'std_error': 0,
                    'sample_size': len(clean_data)
                }
            
            # Calculate basic statistics
            mean = np.mean(clean_data)
            std_error = stats.sem(clean_data)
            
            # Handle case where std_error is 0 or very small
            if std_error < 1e-10:
                return {
                    'mean': mean,
                    'ci_lower': mean,
                    'ci_upper': mean,
                    'std_error': 0,
                    'sample_size': len(clean_data)
                }
            
            # Calculate CI using t-distribution
            t_value = stats.t.ppf((1 + self.confidence_level) / 2, len(clean_data) - 1)
            margin = t_value * std_error
            
            return {
                'mean': mean,
                'ci_lower': mean - margin,
                'ci_upper': mean + margin,
                'std_error': std_error,
                'sample_size': len(clean_data)
            }
            
        except Exception as e:
            print(f"Error calculating confidence interval: {str(e)}")
            return {
                'mean': 0,
                'ci_lower': 0,
                'ci_upper': 0,
                'std_error': 0,
                'sample_size': 0
            }
    
    def analyze_results(self):
        """Analyze results with robust error handling."""
        if not self.simulation_results:
            print("No simulation results to analyze")
            return None
            
        try:
            metrics_to_analyze = defaultdict(list)
            section_metrics = defaultdict(lambda: defaultdict(list))
            
            # Collect metrics across all runs
            for run in self.simulation_results:
                metrics_to_analyze['time_in_system'].append(run['avg_time_in_system'])
                metrics_to_analyze['completion_rate'].append(run['completion_rate'])
                metrics_to_analyze['rejection_rate'].append(run['rejection_rate'])
                
                for section, section_data in run['section_metrics'].items():
                    for metric, value in section_data.items():
                        section_metrics[section][metric].append(value)
            
            # Calculate confidence intervals
            results = {
                'overall_metrics': {},
                'section_metrics': {}
            }
            
            # Overall metrics
            for metric in metrics_to_analyze:
                results['overall_metrics'][metric] = self.calculate_confidence_interval(
                    metrics_to_analyze[metric]
                )
            
            # Section-specific metrics
            for section, metrics in section_metrics.items():
                results['section_metrics'][section] = {}
                for metric, values in metrics.items():
                    results['section_metrics'][section][metric] = self.calculate_confidence_interval(
                        values
                    )
            
            return results
            
        except Exception as e:
            print(f"Error analyzing results: {str(e)}")
            return None
    
    def print_analysis(self, results):
        """Print analysis results with additional information."""
        if not results:
            print("No results to display")
            return
            
        try:
            print(f"\nConfidence Interval Analysis ({self.confidence_level*100}% confidence level)")
            print("=" * 80)
            
            # Print overall metrics
            print("\nOverall Metrics:")
            for metric, values in results['overall_metrics'].items():
                print(f"\n{metric.replace('_', ' ').title()}:")
                print(f"  Mean: {values['mean']:.3f}")
                print(f"  Confidence Interval: [{values['ci_lower']:.3f}, {values['ci_upper']:.3f}]")
                print(f"  Standard Error: {values['std_error']:.3f}")
                print(f"  Sample Size: {values['sample_size']}")
            
            # Print section-specific metrics
            print("\nSection-Specific Metrics:")
            for section, metrics in results['section_metrics'].items():
                print(f"\n{section.capitalize()}:")
                for metric, values in metrics.items():
                    print(f"\n  {metric.replace('_', ' ').title()}:")
                    print(f"    Mean: {values['mean']:.3f}")
                    print(f"    Confidence Interval: [{values['ci_lower']:.3f}, {values['ci_upper']:.3f}]")
                    print(f"    Standard Error: {values['std_error']:.3f}")
                    print(f"    Sample Size: {values['sample_size']}")
                    
        except Exception as e:
            print(f"Error printing analysis: {str(e)}")