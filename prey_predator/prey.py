import random
from typing import List
from animal import Animal, Gender

class Prey(Animal):
    """Class representing a prey animal in the simulation"""
    def __init__(self, id: int, gender: Gender, birth_time: float):
        super().__init__(id, gender, birth_time)
        
    @classmethod
    def reproduce(cls, parent_id: int, current_time: float) -> List['Prey']:
        """Create 1-5 new prey with random gender"""
        num_offspring = random.randint(1, 5)
        offspring = []
        
        for i in range(num_offspring):
            gender = random.choice(list(Gender))
            new_id = parent_id * 1000 + i  # Hierarchical ID system
            offspring.append(cls(new_id, gender, current_time))
            
        return offspring