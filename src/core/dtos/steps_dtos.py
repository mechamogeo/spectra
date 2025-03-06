from dataclasses import dataclass

@dataclass
class Steps:
    """Represents the probability and description of a specific analysis step"""
    step_name: str
    description: str
    probability: float
