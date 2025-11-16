import numpy as np
from dataclasses import dataclass

@dataclass
class Phase:
    """A class to describe a balancing phase consisting of  energy packets for excess and deficit"""
    def __init__(self, energy_excess: float, energy_deficit: float, id: int = None):
        self.id = id

        self.starts_excess = np.array([0.])
        self.starts_deficit = np.array([0.])
        self.energy_excess = np.array([energy_excess])
        self.energy_deficit = np.array([energy_deficit])
        self.excess_balanced = np.array([False])
        self.deficit_balanced = np.array([False])

        self.excess_ids = np.array([self.id])