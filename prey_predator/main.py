import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm

from animal import Gender
from fes import Event, EventScheduler, EventType
from population_state import PopulationState
from predator import Predator
from prey import Prey

@dataclass
class SimulationParameters:
    """Parameters for the simulation"""
    lambda1: float  # prey reproduction rate
    lambda2: float  # predator reproduction rate
    lambda3: float  # predation rate
    mu1: float     # prey death rate
    mu2: float     # predator death rate
    K1: float      # prey population threshold
    K2: float      # prey/predator ratio threshold
    alpha: float   # prey competition coefficient
    beta: float    # predator competition coefficient

    def to_dict(self) -> dict:
        return{
            'lambda1': self.lambda1,
            'lambda2': self.lambda2,
            'lambda3': self.lambda3,
            'mu1': self.mu1,
            'mu2': self.mu2,
            'K1': self.K1,
            'K2': self.K2,
            'alpha': self.alpha,
            'beta': self.beta
        }

class SimulationStatistics:
    """Collect and analyze simulation statistics"""
    def __init__(self):
        self.times: List[float] = []
        self.predator_counts: List[int] = []
        self.prey_counts: List[int] = []
        self.male_predator_counts: List[int] = []
        self.female_predator_counts: List[int] = []
        self.male_prey_counts: List[int] = []
        self.female_prey_counts: List[int] = []
    
    def record_state(self, time: float, population: PopulationState):
        """Record the current state of the simulation"""
        mp, fp, mr, fr = population.get_counts()
        # Verify all data is being recorded consistently
        self.times.append(time)
        self.male_predator_counts.append(mp)
        self.female_predator_counts.append(fp)
        self.male_prey_counts.append(mr)
        self.female_prey_counts.append(fr)
        self.predator_counts.append(mp + fp)
        self.prey_counts.append(mr + fr)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert statistics to a pandas DataFrame"""
        return pd.DataFrame({
            'time': self.times,
            'predators': self.predator_counts,
            'prey': self.prey_counts,
            'male_predators': self.male_predator_counts,
            'female_predators': self.female_predator_counts,
            'male_prey': self.male_prey_counts,
            'female_prey': self.female_prey_counts
        })
    
class Simulation:
    def __init__(self, params: SimulationParameters, 
                 initial_population: Dict[str, int],
                 end_time: float,
                 statistics_interval: float = 1.0):
        self.params = params
        self.initial_population = initial_population
        self.end_time = end_time
        self.statistics_interval = statistics_interval
    
    def initialize_population(self) -> PopulationState:
        """Create initial population"""
        population = PopulationState()
        
        # Reset ID counters
        Predator.reset_id_counter()
        Prey.reset_id_counter()
        
        # Add initial predators
        for _ in range(self.initial_population.get('male_predators', 0)):
            population.add_animal(Predator(Gender.MALE, 0))
        for _ in range(self.initial_population.get('female_predators', 0)):
            population.add_animal(Predator(Gender.FEMALE, 0))
            
        # Add initial prey
        for _ in range(self.initial_population.get('male_prey', 0)):
            population.add_animal(Prey(Gender.MALE, 0))
        for _ in range(self.initial_population.get('female_prey', 0)):
            population.add_animal(Prey(Gender.FEMALE, 0))
            
        return population
    
    def run_single_simulation(self) -> SimulationStatistics:
        print("\n=== Starting New Simulation ===")
        scheduler = EventScheduler()  
        scheduler.population = self.initialize_population()
        stats = SimulationStatistics()
        
        mp, fp, mr, fr = scheduler.population.get_counts()
        print(f"Initial population - Predators(M/F): {mp}/{fp}, Prey(M/F): {mr}/{fr}")
        
        # Record initial state
        stats.record_state(0, scheduler.population)
        
        print("\nScheduling initial events...")
        scheduler.schedule_reproduction_events(0, self.params.to_dict())
        scheduler.schedule_death_events(0, self.params.to_dict())
        scheduler.schedule_predation_events(0, self.params.to_dict())
        
        next_stats_time = self.statistics_interval
        
        while scheduler.current_time < self.end_time:
            print(f"\nCurrent time: {scheduler.current_time}")
            print(f"population: {scheduler.population.get_counts()}")
            event = scheduler.get_next_event()
            print(f"Next event: {event.event_type} at time {event.time}")
            if event is None:
                print("No more events in queue!")
                break
                
            if event.time >= next_stats_time:
                # Record all stats up to current time
                while next_stats_time <= event.time:
                    stats.record_state(next_stats_time, scheduler.population)
                    next_stats_time += self.statistics_interval
                    
            scheduler.handle_event(event, self.params.to_dict())
            
            # Check if population is extinct
            mp, fp, mr, fr = scheduler.population.get_counts()
            if (mp + fp) == 0:
                print("Predators extinct!")
                stats.record_state(event.time, scheduler.population)
                break
            if (fr + mr) == 0:
                print("Prey extinct!")
                stats.record_state(event.time, scheduler.population)
                break
            if (mp + fp + mr + fr) == 0:
                print("Population extinct!")
                stats.record_state(event.time, scheduler.population)
                break
        
        # Record final state if needed
        if scheduler.current_time < self.end_time:
            stats.record_state(scheduler.current_time, scheduler.population)
        
        return stats
    
    def run_multiple_simulations(self, num_simulations: int) -> List[SimulationStatistics]:
        """Run multiple simulations for confidence intervals"""
        all_stats = []
        for _ in tqdm(range(num_simulations), desc="Running simulations"):
            stats = self.run_single_simulation()
            all_stats.append(stats)
        return all_stats
    
class SimulationAnalyzer:
    """Analyze and visualize simulation results"""
    @staticmethod
    def calculate_confidence_intervals(all_stats: List[SimulationStatistics], 
                                    confidence_level: float = 0.95) -> pd.DataFrame:
        """Calculate confidence intervals for population counts"""
        # Convert all statistics to DataFrames
        dfs = [stats.to_dataframe() for stats in all_stats]
        
        # Combine DataFrames
        combined_df = pd.concat(dfs)
        
        # Group by time and calculate statistics
        grouped = combined_df.groupby('time')
        means = grouped.mean()
        
        # Calculate confidence intervals
        ci_results = {}
        for column in ['predators', 'prey']:
            ci_lower = []
            ci_upper = []
            for time in means.index:
                values = combined_df[combined_df['time'] == time][column]
                ci = stats.t.interval(confidence_level, len(values)-1, 
                                    loc=np.mean(values), 
                                    scale=stats.sem(values))
                ci_lower.append(ci[0])
                ci_upper.append(ci[1])
            
            ci_results[f'{column}_mean'] = means[column]
            ci_results[f'{column}_ci_lower'] = ci_lower
            ci_results[f'{column}_ci_upper'] = ci_upper
            
        return pd.DataFrame(ci_results, index=means.index)
    
    @staticmethod
    def plot_simulation_results(stats: SimulationStatistics, title: str = "Population Dynamics"):
        if not stats.times:
            print("No data to plot!")
            return
            
        plt.figure(figsize=(12, 6))
        
        # Plot population counts
        plt.plot(stats.times, stats.predator_counts, 'r-', label='Predators', linewidth=2)
        plt.plot(stats.times, stats.prey_counts, 'b-', label='Prey', linewidth=2)
        
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Plot gender ratios
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(stats.times, stats.male_predator_counts, 'r--', label='Male Predators')
        plt.plot(stats.times, stats.female_predator_counts, 'r:', label='Female Predators')
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.title('Predator Gender Distribution')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(stats.times, stats.male_prey_counts, 'b--', label='Male Prey')
        plt.plot(stats.times, stats.female_prey_counts, 'b:', label='Female Prey')
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.title('Prey Gender Distribution')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_population_ratio(stats: SimulationStatistics, title: str = "Population Ratio Analysis"):
        """Plot the prey-to-predator ratio over time"""
        ratios = [p/pr if pr > 0 else float('inf') 
                 for p, pr in zip(stats.prey_counts, stats.predator_counts)]
        
        plt.figure(figsize=(12, 6))
        plt.plot(stats.times, ratios, 'g-', label='Prey/Predator Ratio')
        plt.xlabel('Time')
        plt.ylabel('Ratio')
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def plot_phase_space(stats: SimulationStatistics, title: str = "Phase Space Analysis"):
        """Create a phase space plot of predator vs prey populations with trajectory lines"""
        plt.figure(figsize=(10, 10))
        
        # Plot the trajectory line to show the evolution of the system
        plt.plot(stats.predator_counts, stats.prey_counts, 'b-', alpha=0.5, 
                label='Population Trajectory')
        
        # Plot points with reduced size and transparency
        plt.scatter(stats.predator_counts, stats.prey_counts, c='k', s=20, alpha=0.3)
        
        # Add arrows to show direction of evolution
        for i in range(0, len(stats.times)-1, 20):  # Add arrows every 20 points
            if i+1 < len(stats.predator_counts):
                plt.arrow(stats.predator_counts[i], stats.prey_counts[i],
                        (stats.predator_counts[i+1] - stats.predator_counts[i])*0.2,
                        (stats.prey_counts[i+1] - stats.prey_counts[i])*0.2,
                        head_width=1, head_length=1, fc='r', ec='r', alpha=0.5)
        
        plt.xlabel('Predator Population')
        plt.ylabel('Prey Population')
        plt.title(title)
        plt.grid(True)
        plt.legend()
        
        # Set reasonable axis limits based on data
        max_pred = max(stats.predator_counts)
        max_prey = max(stats.prey_counts)
        plt.xlim(0, max_pred * 1.1)
        plt.ylim(0, max_prey * 1.1)
        
        plt.show()

    @staticmethod
    def plot_population_events(event_records: List[Event], title: str = "Population Events Analysis"):
        """Plot frequency of different population events over time"""
        events_df = pd.DataFrame(event_records)
        event_counts = events_df.groupby(['time', 'event_type']).size().unstack()
        
        plt.figure(figsize=(12, 6))
        for event_type in EventType:
            if event_type.name in event_counts.columns:
                plt.plot(event_counts.index, event_counts[event_type.name], 
                        label=event_type.name)
        
        plt.xlabel('Time')
        plt.ylabel('Event Frequency')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    def analyze_complete_simulation(stats: SimulationStatistics, title_prefix: str = "Simulation Analysis"):
        """Run a complete analysis with all available plots"""
        # Original population dynamics plot
        SimulationAnalyzer.plot_simulation_results(stats, f"{title_prefix} - Basic Population Dynamics")
        
        # New ratio analysis
        SimulationAnalyzer.plot_population_ratio(stats, f"{title_prefix} - Population Ratio")
        
        # Phase space analysis
        SimulationAnalyzer.plot_phase_space(stats, f"{title_prefix} - Phase Space")
        
        # Add population events plot if event data is available
        if hasattr(stats, 'event_records'):
            SimulationAnalyzer.plot_population_events(stats.event_records, f"{title_prefix} - Event Analysis")
        
    @staticmethod
    def plot_confidence_intervals(ci_data: pd.DataFrame, title: str = "Population Dynamics with Confidence Intervals"):
        """Plot population means with confidence intervals"""
        plt.figure(figsize=(12, 6))
        
        # Plot predator confidence intervals
        plt.fill_between(ci_data.index, 
                        ci_data['predators_ci_lower'],
                        ci_data['predators_ci_upper'],
                        color='r', alpha=0.2)
        plt.plot(ci_data.index, ci_data['predators_mean'], 'r-', label='Predators')
        
        # Plot prey confidence intervals
        plt.fill_between(ci_data.index,
                        ci_data['prey_ci_lower'],
                        ci_data['prey_ci_upper'],
                        color='b', alpha=0.2)
        plt.plot(ci_data.index, ci_data['prey_mean'], 'b-', label='Prey')
        
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()