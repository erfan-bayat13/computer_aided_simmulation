from typing import List, Tuple

from animal import Animal
from predator import Predator
from prey import Prey


class PopulationState:
    def __init__(self):
        self.male_predators: List[Predator] = []
        self.female_predators: List[Predator] = []
        self.male_prey: List[Prey] = []
        self.female_prey: List[Prey] = []
    
    def add_animal(self, animal: Animal):
        """Add an animal to the appropriate list based on its type and gender"""
        if isinstance(animal, Prey):
            if animal.is_male():
                self.male_prey.append(animal)
            else:
                self.female_prey.append(animal)
        else:  # Predator
            if animal.is_male():
                self.male_predators.append(animal)
            else:
                self.female_predators.append(animal)

    def remove_animal(self, animal: Animal) -> bool:
        """Remove an animal from the appropriate list"""
        try:
            if isinstance(animal, Prey):
                if animal.is_male():
                    self.male_prey.remove(animal)
                else:
                    self.female_prey.remove(animal)
            else:  # Predator
                if animal.is_male():
                    self.male_predators.remove(animal)
                else:
                    self.female_predators.remove(animal)
            return True
        except ValueError:
            return False
    
    def get_counts(self) -> Tuple[int, int, int, int]:
        """Return current population counts (MP, FP, MR, FR)"""
        return (
            len(self.male_predators),
            len(self.female_predators),
            len(self.male_prey),
            len(self.female_prey)
        )

