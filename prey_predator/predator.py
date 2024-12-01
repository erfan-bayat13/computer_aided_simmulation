import random
from typing import List
from animal import Animal, Gender

class Predator(Animal):
    """Class representing a predator animal in the simulation"""
    _next_id = 1  # Static counter for IDs
    
    def __init__(self, gender: Gender, birth_time: float):
        super().__init__(Predator._next_id, gender, birth_time)
        Predator._next_id += 1
        
    @classmethod
    def reproduce(cls, current_time: float) -> List['Predator']:
        """Create 1-5 new predators with random gender"""
        num_offspring = random.randint(1, 3)
        offspring = []
        
        for _ in range(num_offspring):
            gender = random.choice(list(Gender))
            offspring.append(cls(gender, current_time))
            
        return offspring

    @classmethod
    def reset_id_counter(cls):
        """Reset the ID counter to 1"""
        cls._next_id = 1
