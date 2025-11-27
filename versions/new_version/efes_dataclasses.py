import numpy as np
from numba import int32, float64, boolean

from numba.experimental import jitclass

spec = [
    ('id', int32),

    ('capacity_excess', int32),
    ('capacity_deficit', int32),
    ('size_excess', int32),
    ('size_deficit', int32),

    ('starts_excess', float64[:]),
    ('starts_deficit', float64[:]),
    ('energy_excess', float64[:]),
    ('energy_deficit', float64[:]),
    ('excess_ids', int32[:]),
]
@jitclass(spec)
class Phase:
    """A class to describe a balancing phase consisting of  energy packets for excess and deficit"""
    def __init__(self, energy_excess: float, energy_deficit: float, id: int,  initial_capacity):
        self.id = id

        self.capacity_excess = initial_capacity
        self.capacity_deficit = initial_capacity
        self.size_excess = 1
        self.size_deficit = 1

        self.starts_excess = np.empty(initial_capacity, dtype=np.float64)
        self.starts_deficit = np.empty(initial_capacity, dtype=np.float64)
        self.energy_excess = np.empty(initial_capacity, dtype=np.float64)
        self.energy_deficit = np.empty(initial_capacity, dtype=np.float64)
        self.excess_ids = np.empty(initial_capacity, dtype=np.int32)

        self.starts_excess[0] = 0.
        self.starts_deficit[0] = 0.
        self.energy_excess[0] = energy_excess
        self.energy_deficit[0] = energy_deficit
        self.excess_ids[0] = self.id

    def append_excess(self, excess_start, excess_content, excess_id):
        if self.size_excess >= self.capacity_excess:
            self.capacity_excess *= 2
            self.starts_excess = np.resize(self.starts_excess, self.capacity_excess)
            self.energy_excess = np.resize(self.energy_excess, self.capacity_excess)
            self.excess_ids = np.resize(self.excess_ids, self.capacity_excess)

        self.starts_excess[self.size_excess] = excess_start
        self.energy_excess[self.size_excess] = excess_content

        self.excess_ids[self.size_excess] = excess_id
        self.size_excess += 1

    def append_deficit(self, new_start, energy_remaining):
        if self.size_deficit >= self.capacity_deficit:
            self.capacity_deficit *= 2
            self.starts_deficit = np.resize(self.starts_deficit, self.capacity_deficit)
            self.energy_deficit = np.resize(self.energy_deficit, self.capacity_deficit)

        self.starts_deficit[self.size_deficit] = new_start
        self.energy_deficit[self.size_deficit] = energy_remaining
        self.size_deficit += 1

    def remove_excess(self, index_to_remove, invalid_value=0):

        if index_to_remove < 0:
            index_to_remove += self.size_excess
        if index_to_remove not in (self.size_excess - 1, self.size_excess - 2):
            raise IndexError(
                f"Index {index_to_remove} out of valid range (only last and second-last are allowed)."
            )

        last_idx = self.size_excess - 1

        if index_to_remove == last_idx - 1:
            self.energy_excess[index_to_remove] = self.energy_excess[last_idx]
            self.starts_excess[index_to_remove] = self.starts_excess[last_idx]
            self.excess_ids[index_to_remove] = self.excess_ids[last_idx]

        self.energy_excess[last_idx] = invalid_value
        self.starts_excess[last_idx] = invalid_value
        self.excess_ids[last_idx] = invalid_value

        self.size_excess -= 1

    def get_energy_excess_all(self):
        return self.energy_excess[:self.size_excess]

    def get_energy_excess(self, idx):

        if idx < 0:
            idx = self.size_excess + idx

        if idx < 0 or idx >= self.size_excess:
            raise IndexError("energy_excess index out of range")

        return self.energy_excess[idx]

    def get_starts_excess_all(self):
        return self.starts_excess[:self.size_excess]

    def get_starts_excess(self, idx):

        if idx < 0:
            idx = self.size_excess + idx

        if idx < 0 or idx >= self.size_excess:
            raise IndexError("energy_excess index out of range")

        return self.starts_excess[idx]

    def get_excess_ids_all(self):
        return self.excess_ids[:self.size_excess]

    def get_excess_id(self, idx):

        if idx < 0:
            idx = self.size_excess + idx

        if idx < 0 or idx >= self.size_excess:
            raise IndexError("energy_excess index out of range")

        return self.excess_ids[idx]

    def get_energy_deficit_all(self):
        return self.energy_deficit[:self.size_deficit]

    def get_energy_deficit(self, idx):
        if idx < 0:
            idx = self.size_deficit + idx
        if idx < 0 or idx >= self.size_deficit:
            raise IndexError("energy_deficit index out of range")
        return self.energy_deficit[idx]