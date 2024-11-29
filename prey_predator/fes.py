from dataclasses import dataclass
import enum
import heapq
import random
from typing import List, Optional

from animal import Animal
from population_state import PopulationState
from predator import Predator
from prey import Prey


class EventType(enum):
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
        # Add a random small value to break ties in event times
        self.time = self.time + random.random() * 1e-10

class EventScheduler:
    def __init__(self):
        self.current_time: float = 0
        self.future_events: List[Event] = []
        self.population: PopulationState = PopulationState()
    
    def schedule_event(self, event: Event):
        """Add a new event to the FES"""
        heapq.heappush(self.future_events, event)
    
    def get_next_event(self) -> Event:
        """Get the next event from the FES"""
        if self.future_events:
            return heapq.heappop(self.future_events)
        else:
            return None
    
    def schedule_reproduction_events(self, current_time: float, rates: dict):
        """Schedule reproduction events for all animals in the population"""
        mp, fp, mr, fr = self.population.get_counts()

        # Schedule prey reproduction events
        prey_reproduction_rate = rates['lambda1'] * mr * fr
        if prey_reproduction_rate > 0:
            next_prey_reproduction = current_time + random.expovariate(prey_reproduction_rate)

            if self.population.female_prey:
                female = random.choice(self.population.female_prey)
                self.schedule_event(Event(
                    time=next_prey_reproduction,
                    event_type=EventType.PREY_REPRODUCTION,
                    subject=female
                ))
        
        predator_reproduction_rate = rates['lambda2'] * mp * fp
        if predator_reproduction_rate > 0:
            next_predator_reproduction = current_time + random.expovariate(predator_reproduction_rate)

            if self.population.female_predators:
                female = random.choice(self.population.female_predators)
                self.schedule_event(Event(
                    time=next_predator_reproduction,
                    event_type=EventType.PREDATOR_REPRODUCTION,
                    subject= female
                ))
    
    def schedule_death_events(self,current_time: float, rates: dict):
        """Schedule natural death events based on current population"""
        mp, fp, mr, fr = self.population.get_counts()
        total_prey = mr + fr
        total_predators = mp + fp

        # Calculate modified death rates based on competition effects
        prey_death_rate = rates['mu1'] * total_prey
        if total_prey > rates['K1']:
            prey_death_rate *= (1 + rates['alpha'] * (total_prey/rates['K1'] - 1))
            
        predator_death_rate = rates['mu2'] * total_predators
        if total_prey > 0 and (total_prey/total_predators) < rates['K2']:
            predator_death_rate *= (1 + rates['beta'] * (rates['K2']/(total_prey/total_predators)))
        
        # Schedule individual death events
        for prey in self.population.male_prey + self.population.female_prey:
            death_time = current_time + random.expovariate(prey_death_rate/total_prey)
            self.schedule_event(Event(
                time=death_time,
                event_type=EventType.PREY_DEATH,
                subject=prey
            ))
        
        for predator in self.population.male_predators + self.population.female_predators:
            death_time = current_time + random.expovariate(predator_death_rate/total_predators)
            self.schedule_event(Event(
                time=death_time,
                event_type=EventType.PREDATOR_DEATH,
                subject=predator
            ))

    def schedule_predation_events(self, current_time: float, rates: dict):
        """Schedule predation events"""
        mp, fp, mr, fr = self.population.get_counts()
        total_predators = mp + fp
        total_prey = mr + fr
        
        predation_rate = rates['lambda3'] * total_predators * total_prey
        if predation_rate > 0:
            next_predation = current_time + random.expovariate(predation_rate)
            # Randomly select a predator and prey
            if total_predators > 0 and total_prey > 0:
                predator = random.choice(self.population.male_predators + self.population.female_predators)
                prey = random.choice(self.population.male_prey + self.population.female_prey)
                self.schedule_event(Event(
                    time=next_predation,
                    event_type=EventType.PREDATION,
                    subject=predator,
                    partner=prey
                ))
    
    def handle_events(self,event: Event, rates: dict):
        """Process an event and update the system state"""
        self.current_time = event.time

        if event.event_type == EventType.PREY_REPRODUCTION:
            if event.subject in (self.population.female_prey):
                offspring = Prey.reproduce(event.subject.id, self.current_time)
                for baby in offspring:
                    self.population.add_animal(baby)
        
        elif event.event_type == EventType.PREDATOR_REPRODUCTION:
            if event.subject in (self.population.female_predators):
                offspring = Predator.reproduce(event.subject.id, self.current_time)
                for baby in offspring:
                    self.population.add_animal(baby)
        
        elif event.event_type == EventType.PREDATION:
            if (event.subject in (self.population.male_predators + self.population.female_predators) and 
                event.partner in (self.population.male_prey + self.population.female_prey)):
                self.population.remove_animal(event.partner)
        
        elif event.event_type == EventType.PREY_DEATH:
            self.population.remove_animal(event.subject)
        
        elif event.event_type == EventType.PREDATOR_DEATH:
            self.population.remove_animal(event.subject)
        
        # Schedule new events after state update
        self.schedule_reproduction_events(self.current_time, rates)
        self.schedule_death_events(self.current_time, rates)
        self.schedule_predation_events(self.current_time, rates)
        