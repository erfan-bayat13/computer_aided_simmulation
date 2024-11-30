from dataclasses import dataclass
from enum import Enum
import heapq
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
        # Add a random small value to break ties in event times
        self.time = self.time + random.random() * 1e-10

class EventScheduler:
    def __init__(self, end_time: float):
        self.end_time = end_time
        self.current_time: float = 0
        self.future_events: List[Event] = []
        self.population: PopulationState = PopulationState()
    
    def schedule_event(self, event: Event):
        """Add a new event to the FES"""
        print(f"\nScheduling {event.event_type} at time {event.time:.2f}")
        if event.subject:
            print(f"Subject ID: {event.subject.id}")
        if event.partner:
            print(f"Partner ID: {event.partner.id}")
        heapq.heappush(self.future_events, event)
        print(f"Total events in queue: {len(self.future_events)}")
    
    def get_next_event(self) -> Event:
        """Get the next event from the FES"""
        if self.future_events:
            return heapq.heappop(self.future_events)
        else:
            return None
    
    def schedule_reproduction_events(self, current_time: float, rates: dict):
        """Schedule reproduction events with carrying capacity limits"""
        mp, fp, mr, fr = self.population.get_counts()
        total_prey = mr + fr
        total_predators = mp + fp
        print(f"\n=== Scheduling reproduction at {current_time:.2f} ===")
        print(f"Population - Predators: {total_predators}, Prey: {total_prey}")

        # Schedule prey reproduction only if below carrying capacity
        if mr > 0 and fr > 0 and total_prey < rates['K1']:
            prey_rate = rates['lambda1'] * min(mr, fr)  # Use minimum to prevent explosion
            # Reduce reproduction as approaching K1
            if total_prey > rates['K1'] * 0.8:  # Start slowing at 80% of K1
                prey_rate *= (rates['K1'] - total_prey)/(rates['K1'] * 0.2)
            
            next_time = current_time + random.expovariate(prey_rate)
            if next_time <= self.end_time:
                female = random.choice(self.population.female_prey)
                print(f"Scheduling prey reproduction at t={next_time:.2f}")
                self.schedule_event(Event(
                    time=next_time,
                    event_type=EventType.PREY_REPRODUCTION,
                    subject=female
                ))

        # Schedule predator reproduction only if enough prey
        if mp > 0 and fp > 0 and total_prey > 0:
            current_ratio = total_prey / total_predators
            if current_ratio >= rates['K2']:  # Only reproduce if enough prey
                pred_rate = rates['lambda2'] * min(mp, fp)
                # Reduce reproduction as ratio approaches K2
                if current_ratio < rates['K2'] * 1.2:  # Slow reproduction near K2
                    pred_rate *= (current_ratio - rates['K2'])/(rates['K2'] * 0.2)
                
                next_time = current_time + random.expovariate(pred_rate)
                if next_time <= self.end_time:
                    female = random.choice(self.population.female_predators)
                    print(f"Scheduling predator reproduction at t={next_time:.2f}")
                    self.schedule_event(Event(
                        time=next_time,
                        event_type=EventType.PREDATOR_REPRODUCTION,
                        subject=female
                    ))

    def schedule_death_events(self, current_time: float, rates: dict):
        """Schedule death events based on current population"""
        mp, fp, mr, fr = self.population.get_counts()
        total_prey = mr + fr
        total_predators = mp + fp
        print(f"\n=== Scheduling deaths at {current_time:.2f} ===")
        print(f"Population - Predators: {total_predators}, Prey: {total_prey}")
        print(f"Thresholds - K1 (prey max): {rates['K1']}, K2 (prey/pred ratio): {rates['K2']}")

        # Schedule predator deaths first - they should die quickly with no prey
        if total_predators > 0:
            # Base predator death rate
            predator_death_rate = rates['mu2'] * total_predators
            print(f"Base predator death rate: {predator_death_rate:.3f}")

            # Calculate current ratio outside conditionals
            current_ratio = total_prey / total_predators if total_prey > 0 else 0
            print(f"Current prey/predator ratio: {current_ratio:.3f}")

            if total_prey == 0:
                # Maximum starvation rate when no prey
                competition_factor = 10 + rates['beta'] * rates['K2']  
                predator_death_rate *= competition_factor
                print(f"No prey! Severe starvation factor: {competition_factor:.3f}")
                print(f"Modified predator death rate: {predator_death_rate:.3f}")
            elif current_ratio < rates['K2']:
                # Stronger competition factor when ratio is too low
                competition_factor = 1 + rates['beta'] * (rates['K2']/current_ratio)**2
                predator_death_rate *= competition_factor
                print(f"Ratio below K2. Competition factor: {competition_factor:.3f}")
                print(f"Modified predator death rate: {predator_death_rate:.3f}")

            # Schedule more frequent predator deaths when starving
            num_events = 1 if current_ratio >= rates['K2'] else min(5, total_predators)
            for _ in range(num_events):
                next_time = current_time + random.expovariate(predator_death_rate/num_events)
                if next_time <= self.end_time:
                    predator = random.choice(self.population.male_predators + self.population.female_predators)
                    print(f"Scheduling predator death at t={next_time:.2f}")
                    self.schedule_event(Event(
                        time=next_time,
                        event_type=EventType.PREDATOR_DEATH,
                        subject=predator
                    ))

        # Schedule prey deaths
        if total_prey > 0:
            # Base prey death rate
            prey_death_rate = rates['mu1'] * total_prey
            print(f"Base prey death rate: {prey_death_rate:.3f}")

            # Check prey carrying capacity (K1)
            if total_prey > rates['K1']:
                competition_factor = 1 + rates['alpha'] * (total_prey/rates['K1'] - 1)
                prey_death_rate *= competition_factor
                print(f"Prey exceeds K1. Competition factor: {competition_factor:.3f}")
                print(f"Modified prey death rate: {prey_death_rate:.3f}")

            next_time = current_time + random.expovariate(prey_death_rate)
            if next_time <= self.end_time:
                prey = random.choice(self.population.male_prey + self.population.female_prey)
                print(f"Scheduling prey death at t={next_time:.2f}")
                self.schedule_event(Event(
                    time=next_time,
                    event_type=EventType.PREY_DEATH,
                    subject=prey
                ))

    def schedule_predation_events(self, current_time: float, rates: dict):
        """Schedule single predation event"""
        mp, fp, mr, fr = self.population.get_counts()
        total_predators = mp + fp
        total_prey = mr + fr
        print(f"\n=== Scheduling predation at {current_time:.2f} | Pred: {total_predators}, Prey: {total_prey} ===")

        if total_predators > 0 and total_prey > 0:
            pred_rate = rates['lambda3'] * total_predators * total_prey
            next_time = current_time + random.expovariate(pred_rate)
            if next_time <= self.end_time:
                predator = random.choice(self.population.male_predators + self.population.female_predators)
                prey = random.choice(self.population.male_prey + self.population.female_prey)
                print(f"Scheduling predation for t={next_time:.2f}")
                self.schedule_event(Event(
                    time=next_time,
                    event_type=EventType.PREDATION,
                    subject=predator,
                    partner=prey
                ))

    def handle_event(self, event: Event, rates: dict):
        """Process an event and update system state"""
        print(f"\n=== Handling {event.event_type} at t={event.time:.2f} ===")
        self.current_time = event.time  # Update current time to event time

        # Store initial population
        mp, fp, mr, fr = self.population.get_counts()
        print(f"Population before - Pred(M/F): {mp}/{fp}, Prey(M/F): {mr}/{fr}")

        # Process the event
        if event.event_type == EventType.PREY_REPRODUCTION:
            if event.subject in self.population.female_prey:
                offspring = Prey.reproduce(event.subject.id, self.current_time)
                print(f"Adding {len(offspring)} prey offspring")
                for baby in offspring:
                    self.population.add_animal(baby)
        
        elif event.event_type == EventType.PREDATOR_REPRODUCTION:
            if event.subject in self.population.female_predators:
                offspring = Predator.reproduce(event.subject.id, self.current_time)
                print(f"Adding {len(offspring)} predator offspring")
                for baby in offspring:
                    self.population.add_animal(baby)
        
        elif event.event_type == EventType.PREDATION:
            if (event.subject in (self.population.male_predators + self.population.female_predators) and 
                event.partner in (self.population.male_prey + self.population.female_prey)):
                print(f"Removing prey {event.partner.id}")
                self.population.remove_animal(event.partner)
        
        elif event.event_type == EventType.PREY_DEATH:
            if event.subject in (self.population.male_prey + self.population.female_prey):
                print(f"Removing prey {event.subject.id}")
                self.population.remove_animal(event.subject)
        
        elif event.event_type == EventType.PREDATOR_DEATH:
            if event.subject in (self.population.male_predators + self.population.female_predators):
                print(f"Removing predator {event.subject.id}")
                self.population.remove_animal(event.subject)

        # Print final population
        mp, fp, mr, fr = self.population.get_counts()
        print(f"Population after - Pred(M/F): {mp}/{fp}, Prey(M/F): {mr}/{fr}")

        # Schedule next events
        if mp + fp + mr + fr > 0:  # Only if population exists
            self.schedule_reproduction_events(self.current_time, rates)
            self.schedule_death_events(self.current_time, rates)
            self.schedule_predation_events(self.current_time, rates)