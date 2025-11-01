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
    def __eq__(self, other):
        if not isinstance(other, Phase):
            return NotImplemented
        return (np.array_equal(self.starts_excess, other.starts_excess) and
                np.array_equal(self.starts_deficit, other.starts_deficit) and
                np.array_equal(self.energy_excess, other.energy_excess) and
                np.array_equal(self.energy_deficit, other.energy_deficit) and
                np.array_equal(self.excess_balanced, other.excess_balanced) and
                np.array_equal(self.deficit_balanced, other.deficit_balanced))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = f'Phase {self.id}:\n'
        s += f'starts_excess={self.starts_excess.__str__()}, energy_excess={self.energy_excess.__str__()}, excess_balanced={self.excess_balanced.__str__()}, excess_ids={self.excess_ids.__str__()}\n'
        s += f'starts_deficit={self.starts_deficit.__str__()}, energy_deficit={self.energy_deficit.__str__()}, deficit_balanced={self.deficit_balanced.__str__()}\n'
        return s