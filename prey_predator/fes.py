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
        self.PREY_CRITICAL_THRESHOLD = 20   # Critical level - maximum reproduction boost
        self.PREY_LOW_THRESHOLD = 50       # Low population - increased reproduction
        self.PREY_OPTIMAL_THRESHOLD = 150  # Optimal population - normal rates
        self.PREY_HIGH_THRESHOLD = 300     # High population - decreased reproduction
        
        self.PREDATOR_CRITICAL_THRESHOLD = 3  # Critical level - maximum reproduction boost
        self.PREDATOR_LOW_THRESHOLD = 10      # Low population - increased reproduction
        self.PREDATOR_OPTIMAL_THRESHOLD = 25  # Optimal population - normal rates
        self.PREDATOR_HIGH_THRESHOLD = 75    # High population - decreased reproduction
        
        # Carrying capacity ratio (prey per predator for sustainable ecosystem)
        self.OPTIMAL_PREY_PREDATOR_RATIO = 4.0
    
    def schedule_event(self, event: Event):
        heapq.heappush(self.future_events, event)
    
    def get_next_event(self) -> Optional[Event]:
        return heapq.heappop(self.future_events) if self.future_events else None

    def _create_event(self, time: float, event_type: EventType, subject: Animal, partner: Optional[Animal] = None) -> Event:
        return Event(time=time, event_type=event_type, subject=subject, partner=partner)
    
    def _sigmoid(self, x: float, k: float = 1.0) -> float:
        """Sigmoid function for smooth transitions"""
        return 1 / (1 + math.exp(-k * x))

    def _calculate_prey_reproduction_modifier(self, total_prey: int, total_predators: int) -> float:
        """Calculate realistic modifier for prey reproduction rate"""
        # Base modifier based on population size
        if total_prey <= self.PREY_CRITICAL_THRESHOLD:
            # Maximum reproduction rate at critical levels
            base_modifier = 3.0
        elif total_prey <= self.PREY_LOW_THRESHOLD:
            # Gradually decrease from 3.0 to 1.5 as population increases
            ratio = (total_prey - self.PREY_CRITICAL_THRESHOLD) / (self.PREY_LOW_THRESHOLD - self.PREY_CRITICAL_THRESHOLD)
            base_modifier = 3.0 - (1.5 * ratio)
        elif total_prey <= self.PREY_OPTIMAL_THRESHOLD:
            # Normal reproduction rate
            base_modifier = 1.0
        else:
            # Decrease reproduction as population exceeds optimal threshold
            excess_ratio = (total_prey - self.PREY_OPTIMAL_THRESHOLD) / (self.PREY_HIGH_THRESHOLD - self.PREY_OPTIMAL_THRESHOLD)
            base_modifier = max(0.2, 1.0 - (0.8 * self._sigmoid(excess_ratio - 0.5, k=4)))
        
        # Adjust based on predator pressure
        if total_predators > 0:
            prey_predator_ratio = total_prey / total_predators
            if prey_predator_ratio < self.OPTIMAL_PREY_PREDATOR_RATIO:
                # Increase reproduction when there are too many predators
                predator_pressure_modifier = 1.0 + 0.5 * (1 - prey_predator_ratio / self.OPTIMAL_PREY_PREDATOR_RATIO)
                base_modifier *= predator_pressure_modifier
        
        return base_modifier

    def _calculate_predator_reproduction_modifier(self, total_predators: int, total_prey: int) -> float:
        """Calculate realistic modifier for predator reproduction rate"""
        # Base modifier based on population size
        if total_predators <= self.PREDATOR_CRITICAL_THRESHOLD:
            # Maximum reproduction rate at critical levels
            base_modifier = 2.5
        elif total_predators <= self.PREDATOR_LOW_THRESHOLD:
            # Gradually decrease from 2.5 to 1.3 as population increases
            ratio = (total_predators - self.PREDATOR_CRITICAL_THRESHOLD) / (self.PREDATOR_LOW_THRESHOLD - self.PREDATOR_CRITICAL_THRESHOLD)
            base_modifier = 2.5 - (1.2 * ratio)
        elif total_predators <= self.PREDATOR_OPTIMAL_THRESHOLD:
            # Normal reproduction rate
            base_modifier = 1.0
        else:
            # Decrease reproduction as population exceeds optimal threshold
            excess_ratio = (total_predators - self.PREDATOR_OPTIMAL_THRESHOLD) / (self.PREDATOR_HIGH_THRESHOLD - self.PREDATOR_OPTIMAL_THRESHOLD)
            base_modifier = max(0.3, 1.0 - (0.7 * self._sigmoid(excess_ratio - 0.5, k=3)))
        
        # Adjust based on prey availability
        if total_prey > 0:
            prey_predator_ratio = total_prey / total_predators
            if prey_predator_ratio < self.OPTIMAL_PREY_PREDATOR_RATIO:
                # Decrease reproduction when there isn't enough prey
                food_scarcity_modifier = max(0.3, prey_predator_ratio / self.OPTIMAL_PREY_PREDATOR_RATIO)
                base_modifier *= food_scarcity_modifier
            elif prey_predator_ratio > self.OPTIMAL_PREY_PREDATOR_RATIO * 1.5:
                # Slight increase in reproduction when prey is abundant
                food_abundance_modifier = min(1.3, 1.0 + 0.3 * self._sigmoid(prey_predator_ratio / self.OPTIMAL_PREY_PREDATOR_RATIO - 1.5, k=1))
                base_modifier *= food_abundance_modifier
        
        return base_modifier

    def _calculate_predation_modifier(self, total_prey: int, total_predators: int) -> float:
        """Calculate realistic modifier for predation rate"""
        if total_prey <= self.PREY_CRITICAL_THRESHOLD:
            # Significantly reduced predation at critical prey levels
            base_modifier = 0.3
        elif total_prey <= self.PREY_LOW_THRESHOLD:
            # Gradually increase predation as prey population grows
            ratio = (total_prey - self.PREY_CRITICAL_THRESHOLD) / (self.PREY_LOW_THRESHOLD - self.PREY_CRITICAL_THRESHOLD)
            base_modifier = 0.3 + (0.7 * ratio)
        elif total_prey <= self.PREY_OPTIMAL_THRESHOLD:
            # Normal predation rate
            base_modifier = 1.0
        else:
            # Increased predation when prey is abundant
            excess_ratio = (total_prey - self.PREY_OPTIMAL_THRESHOLD) / (self.PREY_HIGH_THRESHOLD - self.PREY_OPTIMAL_THRESHOLD)
            base_modifier = 1.0 + (0.5 * self._sigmoid(excess_ratio, k=2))
        
        # Adjust based on predator competition
        if total_predators > 1:
            prey_per_predator = total_prey / total_predators
            if prey_per_predator < self.OPTIMAL_PREY_PREDATOR_RATIO:
                # Increase predation effort when food is scarce
                competition_modifier = 1.0 + 0.3 * (1 - prey_per_predator / self.OPTIMAL_PREY_PREDATOR_RATIO)
                base_modifier *= competition_modifier
        
        return base_modifier
    

    
    def schedule_prey_reproduction(self, current_time: float, rates: dict):
        """Schedule prey reproduction events with realistic dynamic rates"""
        mr, fr = len(self.population.male_prey), len(self.population.female_prey)
        mp, fp = len(self.population.male_predators), len(self.population.female_predators)
        total_prey = mr + fr
        total_predators = mp + fp
        
        if mr * fr > 0 and self.population.female_prey:
            # Calculate base modifier based on population size
            if total_prey <= self.PREY_CRITICAL_THRESHOLD:
                base_modifier = 3.0
            elif total_prey <= self.PREY_LOW_THRESHOLD:
                ratio = (total_prey - self.PREY_CRITICAL_THRESHOLD) / (self.PREY_LOW_THRESHOLD - self.PREY_CRITICAL_THRESHOLD)
                base_modifier = 3.0 - (1.5 * ratio)
            else:
                base_modifier = 1.0
                # Add population control when prey numbers are high
                if total_prey > self.PREY_OPTIMAL_THRESHOLD:
                    overpopulation_factor = min(1.0, self.PREY_OPTIMAL_THRESHOLD / total_prey)
                    base_modifier *= overpopulation_factor
            
            # Calculate final rate and schedule event
            adjusted_rate = rates['lambda1'] * base_modifier
            # Use square root scaling to prevent quadratic explosion
            breeding_pairs = min(mr, fr)  # Use actual breeding pairs instead of mr * fr
            prey_reproduction_time = current_time + random.expovariate(adjusted_rate * breeding_pairs)
            
            self.schedule_event(self._create_event(
                prey_reproduction_time,
                EventType.PREY_REPRODUCTION,
                random.choice(self.population.female_prey)
            ))

    def schedule_predator_reproduction(self, current_time: float, rates: dict):
        """Schedule predator reproduction events with realistic dynamic rates"""
        mp, fp = len(self.population.male_predators), len(self.population.female_predators)
        mr, fr = len(self.population.male_prey), len(self.population.female_prey)
        total_predators = mp + fp
        total_prey = mr + fr
        
        if mp * fp > 0 and self.population.female_predators:
            # Calculate base modifier based on population size
            if total_predators <= self.PREDATOR_CRITICAL_THRESHOLD:
                # Maximum reproduction rate at critical levels
                base_modifier = 2.5
            elif total_predators <= self.PREDATOR_LOW_THRESHOLD:
                # Gradually decrease from 2.5 to 1.3 as population increases
                ratio = (total_predators - self.PREDATOR_CRITICAL_THRESHOLD) / (self.PREDATOR_LOW_THRESHOLD - self.PREDATOR_CRITICAL_THRESHOLD)
                base_modifier = 2.5 - (1.2 * ratio)
            else:
                # Normal reproduction rate with scaling based on prey availability
                base_modifier = 1.0
            
            # Adjust based on prey availability - important for predator growth
            if total_prey > 0:
                prey_predator_ratio = total_prey / total_predators
                if prey_predator_ratio > self.OPTIMAL_PREY_PREDATOR_RATIO:
                    # Increase reproduction when prey is abundant
                    prey_bonus = min(2.0, prey_predator_ratio / self.OPTIMAL_PREY_PREDATOR_RATIO)
                    base_modifier *= prey_bonus
            
            # Calculate final rate and schedule event
            adjusted_rate = rates['lambda2'] * base_modifier
            # Use square root scaling to prevent quadratic explosion
            breeding_pairs = min(mp, fp)  # Use actual breeding pairs instead of mp * fp
            predator_reproduction_time = current_time + random.expovariate(adjusted_rate * breeding_pairs)
            
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
        """Schedule predation events with realistic dynamic rates"""
        total_predators = len(self.population.male_predators) + len(self.population.female_predators)
        total_prey = len(self.population.male_prey) + len(self.population.female_prey)

        if total_predators > 0 and total_prey > 0:
            modifier = self._calculate_predation_modifier(total_prey, total_predators)
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
        self._last_event_type = event.event_type
        
        # Get population counts before the event
        old_mp, old_fp, old_mr, old_fr = self.population.get_counts()
        old_total_predators = old_mp + old_fp
        old_total_prey = old_mr + old_fr
        
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

        # Get new population counts
        new_mp, new_fp, new_mr, new_fr = self.population.get_counts()
        new_total_predators = new_mp + new_fp
        new_total_prey = new_mr + new_fr
        
        # Calculate population change thresholds
        prey_change_ratio = abs(new_total_prey - old_total_prey) / max(old_total_prey, 1)
        pred_change_ratio = abs(new_total_predators - old_total_predators) / max(old_total_predators, 1)
        
        # Reschedule events if:
        # 1. Population changed significantly (>10% change)
        # 2. Current event was invalid
        # 3. Population is near critical thresholds
        should_reschedule = (
            prey_change_ratio > 0.1 or 
            pred_change_ratio > 0.1 or 
            not event_valid or
            new_total_prey <= self.PREY_CRITICAL_THRESHOLD * 1.5 or
            new_total_predators <= self.PREDATOR_CRITICAL_THRESHOLD * 1.5
        )
        
        if should_reschedule:
            # Clear future events of the same type
            self.future_events = [e for e in self.future_events 
                                if e.event_type != event.event_type]
            # Reschedule this type of event
            if new_total_prey > 0 and new_total_predators > 0:
                if event.event_type == EventType.PREY_REPRODUCTION:
                    self.schedule_prey_reproduction(self.current_time, rates)
                elif event.event_type == EventType.PREDATOR_REPRODUCTION:
                    self.schedule_predator_reproduction(self.current_time, rates)
                elif event.event_type == EventType.PREDATION:
                    self.schedule_predation_events(self.current_time, rates)
                elif event.event_type == EventType.PREY_DEATH:
                    self.schedule_prey_deaths(self.current_time, rates)
                elif event.event_type == EventType.PREDATOR_DEATH:
                    self.schedule_predator_deaths(self.current_time, rates)
        
        # Always schedule new events if population isn't extinct
        elif new_total_prey > 0 and new_total_predators > 0:
            self.schedule_reproduction_events(self.current_time, rates)
            self.schedule_death_events(self.current_time, rates)
            self.schedule_predation_events(self.current_time, rates)