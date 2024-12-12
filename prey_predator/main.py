import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm

from animal import Gender
from fes2 import EventScheduler
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
        return {
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
        self.times.append(time)
        self.male_predator_counts.append(mp)
        self.female_predator_counts.append(fp)
        self.male_prey_counts.append(mr)
        self.female_prey_counts.append(fr)
        self.predator_counts.append(mp + fp)
        self.prey_counts.append(mr + fr)

def run_simulation(params: SimulationParameters, 
                  initial_population: Dict[str, int],
                  end_time: float,
                  statistics_interval: float = 1.0) -> SimulationStatistics:
    """Run a single simulation with the given parameters"""
    
    # Initialize scheduler and statistics
    scheduler = EventScheduler()
    stats = SimulationStatistics()
    
    # Initialize population
    Predator.reset_id_counter()
    Prey.reset_id_counter()
    
    # Add initial predators
    for _ in range(initial_population.get('male_predators', 0)):
        scheduler.population.add_animal(Predator(Gender.MALE, 0))
    for _ in range(initial_population.get('female_predators', 0)):
        scheduler.population.add_animal(Predator(Gender.FEMALE, 0))
        
    # Add initial prey
    for _ in range(initial_population.get('male_prey', 0)):
        scheduler.population.add_animal(Prey(Gender.MALE, 0))
    for _ in range(initial_population.get('female_prey', 0)):
        scheduler.population.add_animal(Prey(Gender.FEMALE, 0))
    
    # Record initial state
    stats.record_state(0, scheduler.population)
    
    # Schedule initial events
    scheduler._schedule_all_events(params.to_dict())
    
    # Track when to record statistics
    next_stats_time = statistics_interval
    
    # Run simulation
    while scheduler.current_time < end_time:
        event = scheduler.get_next_event()
        if event is None:
            break
            
        if event.time >= next_stats_time:
            while next_stats_time <= event.time:
                stats.record_state(next_stats_time, scheduler.population)
                next_stats_time += statistics_interval
                
        scheduler.handle_event(event, params.to_dict())
        
        # Check for extinction
        mp, fp, mr, fr = scheduler.population.get_counts()
        if (mp + fp) == 0 or (mr + fr) == 0:
            stats.record_state(event.time, scheduler.population)
            break
    
    # Record final state if needed
    if scheduler.current_time < end_time:
        stats.record_state(scheduler.current_time, scheduler.population)
    
    return stats

def run_multiple_simulations(params: SimulationParameters,
                           initial_population: Dict[str, int],
                           end_time: float,
                           num_simulations: int,
                           statistics_interval: float = 1.0) -> List[SimulationStatistics]:
    """Run multiple simulations and return their statistics"""
    all_stats = []
    for _ in tqdm(range(num_simulations), desc="Running simulations"):
        stats = run_simulation(params, initial_population, end_time, statistics_interval)
        all_stats.append(stats)
    return all_stats

def plot_simulation_results(stats: SimulationStatistics, title: str = "Population Dynamics"):
    """Plot the results of a single simulation"""
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

if __name__ == "__main__":
    # Set simulation parameters
    params = SimulationParameters(
        lambda1=0.4,  # prey reproduction rate
        lambda2=0.2,  # predator reproduction rate
        lambda3=0.01, # predation rate
        mu1=0.2,      # prey death rate
        mu2=0.2,      # predator death rate
        K1=200,       # prey population threshold
        K2=4.0,       # prey/predator ratio threshold
        alpha=0.01,   # prey competition coefficient
        beta=0.05     # predator competition coefficient
    )
    
    # Set initial population
    initial_population = {
        'male_predators': 10,
        'female_predators': 10,
        'male_prey': 30,
        'female_prey': 30
    }
    
    # Simple single simulation
    stats = run_simulation(params, initial_population, end_time=100.0)
    plot_simulation_results(stats)