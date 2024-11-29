from dataclasses import dataclass
from enum import Enum


class Gender(Enum):
    MALE = "M"
    FEMALE = "F"

@dataclass
class Animal:
    """Base class for both Predator and Prey"""
    id: int
    gender: Gender
    birth_time: float
    
    def is_female(self) -> bool:
        return self.gender == Gender.FEMALE
    
    def is_male(self) -> bool:
        return self.gender == Gender.MALE