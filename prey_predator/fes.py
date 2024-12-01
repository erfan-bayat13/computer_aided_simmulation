from dataclasses import dataclass
from enum import Enum
import heapq
import math
import random
from typing import List, Optional

from animal import Animal
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
        self.time = self.time + random.random() * 1e-10

class EventScheduler:
    def __init__(self):
        self.current_time: float = 0
        self.future_events: List[Event] = []
        self.population: PopulationState = PopulationState()

        # Population thresholds for rate adjustments
        self.PREY_LOW_THRESHOLD = 75
        self.PREY_HIGH_THRESHOLD = 500
        self.PREDATOR_LOW_THRESHOLD = 5
        self.PREDATOR_HIGH_THRESHOLD = 100
    
    def schedule_event(self, event: Event):
        heapq.heappush(self.future_events, event)
    
    def get_next_event(self) -> Optional[Event]:
        return heapq.heappop(self.future_events) if self.future_events else None

    def _create_event(self, time: float, event_type: EventType, subject: Animal, partner: Optional[Animal] = None) -> Event:
        return Event(time=time, event_type=event_type, subject=subject, partner=partner)
    
    def _calculate_prey_reproduction_modifier(self, total_prey: int) -> float:
        """Calculate modifier for prey reproduction rate based on population"""
        if total_prey <= self.PREY_LOW_THRESHOLD:
            # Increase reproduction rate when population is low
            return 2.0 - (total_prey / self.PREY_LOW_THRESHOLD)
        elif total_prey >= self.PREY_HIGH_THRESHOLD:
            # Decrease reproduction rate when population is high
            return 1.0 / (1.0 + math.log(total_prey / self.PREY_HIGH_THRESHOLD + 1))
        return 1.0
    
    def _calculate_predator_reproduction_modifier(self, total_predators: int) -> float:
        """Calculate modifier for predator reproduction rate based on population"""
        if total_predators <= self.PREDATOR_LOW_THRESHOLD:
            # Increase reproduction rate when population is low
            return 2.0 - (total_predators / self.PREDATOR_LOW_THRESHOLD)
        elif total_predators >= self.PREDATOR_HIGH_THRESHOLD:
            # Decrease reproduction rate when population is high
            return 1.0 / (1.0 + math.log(total_predators / self.PREDATOR_HIGH_THRESHOLD + 1))
        return 1.0
    
    def _calculate_predation_modifier(self, total_prey: int) -> float:
        """Calculate modifier for predation rate based on prey population"""
        if total_prey <= self.PREY_LOW_THRESHOLD:
            # Decrease predation rate when prey is scarce
            return 0.5 + (total_prey / (2 * self.PREY_LOW_THRESHOLD))
        elif total_prey >= self.PREY_HIGH_THRESHOLD:
            # Increase predation rate when prey is abundant
            return 1.0 + math.log(total_prey / self.PREY_HIGH_THRESHOLD + 1)
        return 1.0
    
    
    def schedule_prey_reproduction(self, current_time: float, rates: dict):
        """Schedule prey reproduction events with dynamic rates"""
        mr, fr = len(self.population.male_prey), len(self.population.female_prey)
        total_prey = mr + fr
        
        if mr * fr > 0 and self.population.female_prey:
            modifier = self._calculate_prey_reproduction_modifier(total_prey)
            adjusted_rate = rates['lambda1'] * modifier
            prey_reproduction_time = current_time + random.expovariate(adjusted_rate * mr * fr)
            
            self.schedule_event(self._create_event(
                prey_reproduction_time,
                EventType.PREY_REPRODUCTION,
                random.choice(self.population.female_prey)
            ))

    def schedule_predator_reproduction(self, current_time: float, rates: dict):
        """Schedule predator reproduction events with dynamic rates"""
        mp, fp = len(self.population.male_predators), len(self.population.female_predators)
        total_predators = mp + fp
        
        if mp * fp > 0 and self.population.female_predators:
            modifier = self._calculate_predator_reproduction_modifier(total_predators)
            adjusted_rate = rates['lambda2'] * modifier
            predator_reproduction_time = current_time + random.expovariate(adjusted_rate * mp * fp)
            
            self.schedule_event(self._create_event(
                predator_reproduction_time,
                EventType.PREDATOR_REPRODUCTION,
                random.choice(self.population.female_predators)
            ))

    def schedule_prey_deaths(self, current_time: float, rates: dict):
        """Schedule prey death events"""
        total_prey = len(self.population.male_prey) + len(self.population.female_prey)
        if total_prey > 0:
            prey_death_rate = rates['mu1'] * total_prey
            if total_prey > rates['K1']:
                prey_death_rate *= (1 + rates['alpha'] * (total_prey/rates['K1'] - 1))
            
            for prey in self.population.male_prey + self.population.female_prey:
                death_time = current_time + random.expovariate(prey_death_rate/total_prey)
                self.schedule_event(self._create_event(death_time, EventType.PREY_DEATH, prey))

    def schedule_predator_deaths(self, current_time: float, rates: dict):
        """Schedule predator death events"""
        total_predators = len(self.population.male_predators) + len(self.population.female_predators)
        total_prey = len(self.population.male_prey) + len(self.population.female_prey)
        
        if total_predators > 0:
            predator_death_rate = rates['mu2'] * total_predators
            if total_prey > 0 and (total_prey/total_predators) < rates['K2']:
                predator_death_rate *= (1 + rates['beta'] * (rates['K2']/(total_prey/total_predators)))
            
            for predator in self.population.male_predators + self.population.female_predators:
                death_time = current_time + random.expovariate(predator_death_rate/total_predators)
                self.schedule_event(self._create_event(death_time, EventType.PREDATOR_DEATH, predator))

    def schedule_predation_events(self, current_time: float, rates: dict):
        """Schedule predation events with dynamic rates"""
        total_predators = len(self.population.male_predators) + len(self.population.female_predators)
        total_prey = len(self.population.male_prey) + len(self.population.female_prey)

        if total_predators > 0 and total_prey > 0:
            modifier = self._calculate_predation_modifier(total_prey)
            adjusted_rate = rates['lambda3'] * modifier
            predation_time = current_time + random.expovariate(adjusted_rate * total_predators * total_prey)
            
            predator = random.choice(self.population.male_predators + self.population.female_predators)
            prey = random.choice(self.population.male_prey + self.population.female_prey)
            self.schedule_event(self._create_event(predation_time, EventType.PREDATION, predator, prey))

    # Wrapper functions to maintain compatibility with main.py
    def schedule_reproduction_events(self, current_time: float, rates: dict):
        """Schedule all reproduction events"""
        if event_type := getattr(self, '_last_event_type', None):
            if event_type == EventType.PREY_REPRODUCTION:
                self.schedule_prey_reproduction(current_time, rates)
            elif event_type == EventType.PREDATOR_REPRODUCTION:
                self.schedule_predator_reproduction(current_time, rates)
        else:
            self.schedule_prey_reproduction(current_time, rates)
            self.schedule_predator_reproduction(current_time, rates)

    def schedule_death_events(self, current_time: float, rates: dict):
        """Schedule all death events"""
        if event_type := getattr(self, '_last_event_type', None):
            if event_type == EventType.PREY_DEATH:
                self.schedule_prey_deaths(current_time, rates)
            elif event_type == EventType.PREDATOR_DEATH:
                self.schedule_predator_deaths(current_time, rates)
        else:
            self.schedule_prey_deaths(current_time, rates)
            self.schedule_predator_deaths(current_time, rates)

    def handle_event(self, event: Event, rates: dict):
        """Process an event and update the system state"""
        self.current_time = event.time
        self._last_event_type = event.event_type  # Store the last event type
        
        # Validate event subjects still exist
        all_predators = self.population.male_predators + self.population.female_predators
        all_prey = self.population.male_prey + self.population.female_prey
        
        if not (
            (event.event_type == EventType.PREY_REPRODUCTION and event.subject in self.population.female_prey) or
            (event.event_type == EventType.PREDATOR_REPRODUCTION and event.subject in self.population.female_predators) or
            (event.event_type == EventType.PREDATION and event.subject in all_predators and event.partner in all_prey) or
            (event.event_type == EventType.PREY_DEATH and event.subject in all_prey) or
            (event.event_type == EventType.PREDATOR_DEATH and event.subject in all_predators)
        ):
            return

        # Process valid event
        if event.event_type == EventType.PREY_REPRODUCTION:
            for baby in Prey.reproduce(self.current_time):
                self.population.add_animal(baby)
        elif event.event_type == EventType.PREDATOR_REPRODUCTION:
            for baby in Predator.reproduce(self.current_time):
                self.population.add_animal(baby)
        elif event.event_type == EventType.PREDATION:
            self.population.remove_animal(event.partner)
        elif event.event_type == EventType.PREY_DEATH:
            self.population.remove_animal(event.subject)
        elif event.event_type == EventType.PREDATOR_DEATH:
            self.population.remove_animal(event.subject)

        # Schedule new events if population isn't extinct
        if sum(self.population.get_counts()) > 0:
            self.schedule_reproduction_events(self.current_time, rates)
            self.schedule_death_events(self.current_time, rates)
            self.schedule_predation_events(self.current_time, rates)