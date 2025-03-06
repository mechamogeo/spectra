from dataclasses import dataclass
from typing import List
from .steps_dtos import Steps

@dataclass
class Analisys:
    """ Data Transfer Object for Analisys """
    file_name: str
    steps_probability: List[Steps]
