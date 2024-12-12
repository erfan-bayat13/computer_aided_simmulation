from dataclasses import dataclass
from enum import Enum
import heapq
import random
from typing import List, Optional

from animal import Animal, Gender
from population_state import PopulationState
from predator import Predator
from prey import Prey

class EventType(Enum):
    PREY_REPRODUCTION = "PREY_REPRODUCTION"
    PREDATOR_REPRODUCTION = "PREDATOR_REPRODUCTION" 
    PREDATION = "PREDATION"
    PREY_DEATH = "PREY_DEATH"
    PREDATOR_DEATH = "PREDATOR_DEATH"

@dataclass(order=True)
class Event:
    time: float
    event_type: EventType
    subject: Optional[Animal] = None
    partner: Optional[Animal] = None

    def __post_init__(self):
        # Add tiny random offset to prevent exact time collisions
        self.time = self.time + random.random() * 1e-10

class EventScheduler:
    def __init__(self):
        self.current_time: float = 0
        self.future_events: List[Event] = []
        self.population: PopulationState = PopulationState()
        
        # Population thresholds for rate adjustments
        self.PREY_CRITICAL_THRESHOLD = 20
        self.PREY_LOW_THRESHOLD = 50
        self.PREY_OPTIMAL_THRESHOLD = 150
        self.PREY_HIGH_THRESHOLD = 300
        
        self.PREDATOR_CRITICAL_THRESHOLD = 3
        self.PREDATOR_LOW_THRESHOLD = 10
        self.PREDATOR_OPTIMAL_THRESHOLD = 25
        self.PREDATOR_HIGH_THRESHOLD = 75
        
        self.OPTIMAL_PREY_PREDATOR_RATIO = 4.0
    
    def schedule_event(self, event: Event):
        """Add an event to the priority queue"""
        heapq.heappush(self.future_events, event)
    
    def get_next_event(self) -> Optional[Event]:
        """Get the next event from the priority queue"""
        return heapq.heappop(self.future_events) if self.future_events else None
    
    def _create_event(self, time: float, event_type: EventType, 
                     subject: Optional[Animal] = None, 
                     partner: Optional[Animal] = None) -> Event:
        """Create a new event with the given parameters"""
        return Event(time=time, event_type=event_type, subject=subject, partner=partner)

    def _calculate_base_rates(self, rates: dict, mp: int, fp: int, mr: int, fr: int) -> dict:
        """Calculate base rates based on current population state"""
        total_predators = mp + fp
        total_prey = mr + fr
        
        base_rates = rates.copy()
        
        # Adjust prey reproduction rate based on population
        if total_prey <= self.PREY_CRITICAL_THRESHOLD:
            base_rates['lambda1'] *= 3.0
        elif total_prey <= self.PREY_LOW_THRESHOLD:
            ratio = (total_prey - self.PREY_CRITICAL_THRESHOLD) / (self.PREY_LOW_THRESHOLD - self.PREY_CRITICAL_THRESHOLD)
            base_rates['lambda1'] *= (3.0 - 2.0 * ratio)
        elif total_prey > self.PREY_OPTIMAL_THRESHOLD:
            base_rates['lambda1'] *= max(0.2, self.PREY_OPTIMAL_THRESHOLD / total_prey)

        # Adjust predator reproduction rate based on population and prey availability
        if total_predators > 0:
            prey_pred_ratio = total_prey / total_predators
            if total_predators <= self.PREDATOR_CRITICAL_THRESHOLD:
                base_rates['lambda2'] *= 2.5
            elif total_predators <= self.PREDATOR_LOW_THRESHOLD:
                ratio = (total_predators - self.PREDATOR_CRITICAL_THRESHOLD) / (self.PREDATOR_LOW_THRESHOLD - self.PREDATOR_CRITICAL_THRESHOLD)
                base_rates['lambda2'] *= (2.5 - 1.5 * ratio)
            
            if prey_pred_ratio < self.OPTIMAL_PREY_PREDATOR_RATIO:
                base_rates['lambda2'] *= max(0.3, prey_pred_ratio / self.OPTIMAL_PREY_PREDATOR_RATIO)

        return base_rates

    def _schedule_all_events(self, rates: dict):
        """Schedule all possible events based on current state"""
        mp, fp, mr, fr = self.population.get_counts()
        total_predators = mp + fp
        total_prey = mr + fr
        
        base_rates = self._calculate_base_rates(rates, mp, fp, mr, fr)
        
        # Schedule prey reproduction
        if mr * fr > 0:
            breeding_pairs = min(mr, fr)
            time = self.current_time + random.expovariate(base_rates['lambda1'] * breeding_pairs)
            female = random.choice(self.population.female_prey)
            self.schedule_event(self._create_event(time, EventType.PREY_REPRODUCTION, female))

        # Schedule predator reproduction
        if mp * fp > 0:
            breeding_pairs = min(mp, fp)
            time = self.current_time + random.expovariate(base_rates['lambda2'] * breeding_pairs)
            female = random.choice(self.population.female_predators)
            self.schedule_event(self._create_event(time, EventType.PREDATOR_REPRODUCTION, female))

        # Schedule predation
        if total_predators > 0 and total_prey > 0:
            time = self.current_time + random.expovariate(base_rates['lambda3'] * total_predators * total_prey)
            predator = random.choice(self.population.male_predators + self.population.female_predators)
            prey = random.choice(self.population.male_prey + self.population.female_prey)
            self.schedule_event(self._create_event(time, EventType.PREDATION, predator, prey))

        # Schedule natural deaths
        for prey in self.population.male_prey + self.population.female_prey:
            death_rate = base_rates['mu1'] * (1 + rates['alpha'] * max(0, total_prey/rates['K1'] - 1))
            time = self.current_time + random.expovariate(death_rate)
            self.schedule_event(self._create_event(time, EventType.PREY_DEATH, prey))

        for predator in self.population.male_predators + self.population.female_predators:
            prey_pred_ratio = total_prey/total_predators if total_predators > 0 else 0
            death_rate = base_rates['mu2'] * (1 + rates['beta'] * max(0, rates['K2']/prey_pred_ratio - 1))
            time = self.current_time + random.expovariate(death_rate)
            self.schedule_event(self._create_event(time, EventType.PREDATOR_DEATH, predator))

    def handle_event(self, event: Event, rates: dict):
        """Process an event and update the system state"""
        self.current_time = event.time
        
        # Get current state
        mp, fp, mr, fr = self.population.get_counts()
        total_predators = mp + fp
        total_prey = mr + fr
        
        # Validate event subjects still exist
        all_predators = self.population.male_predators + self.population.female_predators
        all_prey = self.population.male_prey + self.population.female_prey
        
        event_valid = (
            (event.event_type == EventType.PREY_REPRODUCTION and event.subject in self.population.female_prey) or
            (event.event_type == EventType.PREDATOR_REPRODUCTION and event.subject in self.population.female_predators) or
            (event.event_type == EventType.PREDATION and event.subject in all_predators and event.partner in all_prey) or
            (event.event_type == EventType.PREY_DEATH and event.subject in all_prey) or
            (event.event_type == EventType.PREDATOR_DEATH and event.subject in all_predators)
        )

        if event_valid:
            # Process event and update state
            if event.event_type == EventType.PREY_REPRODUCTION:
                for _ in range(random.randint(1, 5)):
                    gender = random.choice(list(Gender))
                    self.population.add_animal(Prey(gender, self.current_time))
            elif event.event_type == EventType.PREDATOR_REPRODUCTION:
                for _ in range(random.randint(1, 3)):
                    gender = random.choice(list(Gender))
                    self.population.add_animal(Predator(gender, self.current_time))
            elif event.event_type == EventType.PREDATION:
                self.population.remove_animal(event.partner)
            elif event.event_type == EventType.PREY_DEATH:
                self.population.remove_animal(event.subject)
            elif event.event_type == EventType.PREDATOR_DEATH:
                self.population.remove_animal(event.subject)

        # Always reschedule all events based on current state only
        self.future_events.clear()
        if total_prey > 0 and total_predators > 0:
            self._schedule_all_events(rates)