import numpy as np
from scipy import stats
from collections import defaultdict
import warnings

class UberConfidenceIntervalAnalyzer:
    """
    Adapted confidence interval analyzer for Uber simulation metrics.
    Handles specific metrics like wait times, ride times, completion rates, etc.
    """
    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level
        self.simulation_results = []
        warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    def add_simulation_run(self, metrics):
        """
        Add results from a single Uber simulation run.
        
        Args:
            metrics (dict): Dictionary containing simulation metrics
        """
        try:
            # Extract and validate key metrics
            run_metrics = {
                'total_clients': metrics['total_clients'],
                'total_matches': metrics['total_matches'],
                'total_completed_rides': metrics['total_completed_rides'],
                'total_cancellations': metrics.get('total_cancellations', 0),
                'completion_rate': (metrics['total_completed_rides'] / metrics['total_clients'] 
                                 if metrics['total_clients'] > 0 else 0),
                'cancellation_rate': (metrics.get('total_cancellations', 0) / metrics['total_clients']
                                   if metrics['total_clients'] > 0 else 0),
            }
            
            # Add wait times if available
            if 'wait_times' in metrics:
                run_metrics['wait_times'] = [t for t in metrics['wait_times'] if t >= 0]
                run_metrics['avg_wait_time'] = np.mean(run_metrics['wait_times']) if run_metrics['wait_times'] else 0
            
            # Add ride times if available
            if 'ride_durations' in metrics:
                run_metrics['ride_times'] = [t for t in metrics['ride_durations'] if t >= 0]
                run_metrics['avg_ride_time'] = np.mean(run_metrics['ride_times']) if run_metrics['ride_times'] else 0
            
            # Add revenue metrics if available
            if 'ride_costs' in metrics:
                run_metrics['ride_costs'] = [c for c in metrics['ride_costs'] if c >= 0]
                run_metrics['avg_ride_cost'] = np.mean(run_metrics['ride_costs']) if run_metrics['ride_costs'] else 0
            
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
            clean_data = self._validate_data(data)
            
            if len(clean_data) < 2:
                return {
                    'mean': np.mean(clean_data) if len(clean_data) == 1 else 0,
                    'ci_lower': 0,
                    'ci_upper': 0,
                    'std_error': 0,
                    'sample_size': len(clean_data),
                    'std_dev': np.std(clean_data) if len(clean_data) > 0 else 0
                }
            
            mean = np.mean(clean_data)
            std_error = stats.sem(clean_data)
            std_dev = np.std(clean_data)
            
            if std_error < 1e-10:
                return {
                    'mean': mean,
                    'ci_lower': mean,
                    'ci_upper': mean,
                    'std_error': 0,
                    'sample_size': len(clean_data),
                    'std_dev': std_dev
                }
            
            t_value = stats.t.ppf((1 + self.confidence_level) / 2, len(clean_data) - 1)
            margin = t_value * std_error
            
            return {
                'mean': mean,
                'ci_lower': mean - margin,
                'ci_upper': mean + margin,
                'std_error': std_error,
                'sample_size': len(clean_data),
                'std_dev': std_dev
            }
            
        except Exception as e:
            print(f"Error calculating confidence interval: {str(e)}")
            return {
                'mean': 0,
                'ci_lower': 0,
                'ci_upper': 0,
                'std_error': 0,
                'sample_size': 0,
                'std_dev': 0
            }
    
    def analyze_results(self):
        """
        Analyze Uber simulation results with comprehensive metrics.
        
        Returns:
            dict: Analysis results including CIs for all metrics
        """
        if not self.simulation_results:
            print("No simulation results to analyze")
            return None
            
        try:
            # Initialize metric collectors
            metrics_to_analyze = defaultdict(list)
            
            # Collect metrics across all runs
            for run in self.simulation_results:
                # Operational metrics
                metrics_to_analyze['completion_rate'].append(run['completion_rate'])
                metrics_to_analyze['cancellation_rate'].append(run['cancellation_rate'])
                metrics_to_analyze['total_clients'].append(run['total_clients'])
                metrics_to_analyze['total_completed_rides'].append(run['total_completed_rides'])
                
                # Time-based metrics
                if 'avg_wait_time' in run:
                    metrics_to_analyze['wait_time'].append(run['avg_wait_time'])
                if 'avg_ride_time' in run:
                    metrics_to_analyze['ride_time'].append(run['avg_ride_time'])
                
                # Financial metrics
                if 'avg_ride_cost' in run:
                    metrics_to_analyze['ride_cost'].append(run['avg_ride_cost'])
            
            # Calculate confidence intervals for all metrics
            results = {}
            for metric, values in metrics_to_analyze.items():
                results[metric] = self.calculate_confidence_interval(values)
            
            return results
            
        except Exception as e:
            print(f"Error analyzing results: {str(e)}")
            return None
    
    def print_analysis(self, results):
        """
        Print comprehensive analysis of Uber simulation results.
        
        Args:
            results (dict): Analysis results from analyze_results()
        """
        if not results:
            print("No results to display")
            return
            
        try:
            print(f"\nUber Simulation Analysis ({self.confidence_level*100}% confidence level)")
            print("=" * 80)
            
            # Group metrics for organized display
            metric_groups = {
                'Operational Metrics': ['completion_rate', 'cancellation_rate', 'total_clients', 'total_completed_rides'],
                'Time Metrics': ['wait_time', 'ride_time'],
                'Financial Metrics': ['ride_cost']
            }
            
            for group_name, metrics in metric_groups.items():
                print(f"\n{group_name}:")
                print("-" * 40)
                
                for metric in metrics:
                    if metric in results:
                        values = results[metric]
                        print(f"\n{metric.replace('_', ' ').title()}:")
                        print(f"  Mean: {values['mean']:.3f}")
                        print(f"  CI: [{values['ci_lower']:.3f}, {values['ci_upper']:.3f}]")
                        print(f"  Std Dev: {values['std_dev']:.3f}")
                        print(f"  Std Error: {values['std_error']:.3f}")
                        print(f"  Sample Size: {values['sample_size']}")
                    
        except Exception as e:
            print(f"Error printing analysis: {str(e)}")