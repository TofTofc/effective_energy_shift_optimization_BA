import numpy as np
from numba import int32, float64, boolean, uint32, uint16, uint8

from numba.experimental import jitclass

spec = [
    ('id', uint32),

    ('capacity_excess', uint16),
    ('capacity_deficit', uint16),
    ('size_excess', uint16),
    ('size_deficit', uint16),
    ('number_of_excess_not_covered' , uint16),

    ('starts_excess', uint32[:]),
    ('starts_deficit', uint32[:]),
    ('energy_excess', uint8[:]),
    ('energy_deficit', uint8[:]),
    ('excess_ids', uint32[:]),
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

        self.starts_excess[0] = 0
        self.starts_deficit[0] = 0
        self.energy_excess[0] = energy_excess
        self.energy_deficit[0] = energy_deficit
        self.excess_ids[0] = self.id

        # Initial Balancing
        if energy_excess > energy_deficit:

            self.energy_excess[0] = energy_deficit

            self.energy_excess[1] = energy_excess - energy_deficit
            self.starts_excess[1] = energy_deficit
            self.excess_ids[1] = self.id

            self.size_excess = 2
            self.number_of_excess_not_covered = 1

        elif energy_deficit > energy_excess:

            self.energy_deficit[0] = energy_excess

            self.energy_deficit[1] = energy_deficit - energy_excess
            self.starts_deficit[1] = energy_excess
            self.size_deficit = 2

            self.number_of_excess_not_covered = 0

        else:
            self.number_of_excess_not_covered = 0

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

    def remove_excess(self, index_to_remove):

        if index_to_remove < 0:
            index_to_remove += self.size_excess

        if index_to_remove < 0 or index_to_remove >= self.size_excess:
            raise IndexError("Index out of range in remove_excess")

        self.starts_excess[index_to_remove:self.size_excess - 1] = self.starts_excess[index_to_remove + 1:self.size_excess]
        self.energy_excess[index_to_remove:self.size_excess - 1] = self.energy_excess[index_to_remove + 1:self.size_excess]
        self.excess_ids[index_to_remove:self.size_excess - 1] = self.excess_ids[index_to_remove + 1:self.size_excess]

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

    def get_starts_deficit_all(self):
        return self.starts_deficit[:self.size_deficit]

    def get_starts_deficit(self, idx):

        if idx < 0:
            idx = self.size_deficit + idx

        if idx < 0 or idx >= self.size_deficit:
            raise IndexError("energy_excess index out of range")

        return self.starts_deficit[idx]

    def set_energy_excess(self, idx, value):
        if idx < 0:
            idx = self.size_excess + idx
        if idx < 0 or idx >= self.size_excess:
            raise IndexError("energy_excess index out of range")
        self.energy_excess[idx] = value

    def set_starts_excess(self, idx, value):
        if idx < 0:
            idx = self.size_excess + idx
        if idx < 0 or idx >= self.size_excess:
            raise IndexError("starts_excess index out of range")
        self.starts_excess[idx] = value

    def set_energy_deficit(self, idx, value):
        if idx < 0:
            idx = self.size_deficit + idx
        if idx < 0 or idx >= self.size_deficit:
            raise IndexError("energy_deficit index out of range")
        self.energy_deficit[idx] = value

    def set_starts_deficit(self, idx, value):
        if idx < 0:
            idx = self.size_deficit + idx
        if idx < 0 or idx >= self.size_deficit:
            raise IndexError("starts_deficit index out of range")
        self.starts_deficit[idx] = value