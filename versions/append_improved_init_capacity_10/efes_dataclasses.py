import numpy as np
from dataclasses import dataclass
@dataclass
class Phase:
    """A class to describe a balancing phase consisting of  energy packets for excess and deficit"""
    def __init__(self, energy_excess: float, energy_deficit: float, id: int = None,  initial_capacity=10):
        self.id = id

        #self.starts_excess = np.array([0.])
        #self.starts_deficit = np.array([0.])
        #self.energy_excess = np.array([energy_excess])
        #self.energy_deficit = np.array([energy_deficit])
        #self.excess_balanced = np.array([False])
        #self.deficit_balanced = np.array([False])

        #self.excess_ids = np.array([self.id])

        self.capacity_excess = initial_capacity
        self.capacity_deficit = initial_capacity
        self.size_excess = 1
        self.size_deficit = 1

        self.starts_excess = np.zeros(initial_capacity, dtype=type(0.))
        self.starts_deficit = np.zeros(initial_capacity, dtype=type(0.))
        self.energy_excess = np.zeros(initial_capacity, dtype=type(energy_excess))
        self.energy_deficit = np.zeros(initial_capacity, dtype=type(energy_deficit))
        self.excess_balanced = np.zeros(initial_capacity, dtype=bool)
        self.deficit_balanced = np.zeros(initial_capacity, dtype=bool)
        self.excess_ids = np.zeros(initial_capacity, dtype=type(id))

        self.starts_excess[0] = 0.
        self.starts_deficit[0] = 0.
        self.energy_excess[0] = energy_excess
        self.energy_deficit[0] = energy_deficit
        self.excess_balanced[0] = False
        self.deficit_balanced[0] = False
        self.excess_ids[0] = self.id

    def append_excess(self, excess_start, excess_content, excess_balanced, excess_id):
        if self.size_excess >= self.capacity_excess:
            self.capacity_excess *= 2
            self.starts_excess = np.resize(self.starts_excess, self.capacity_excess)
            self.energy_excess = np.resize(self.energy_excess, self.capacity_excess)
            self.excess_balanced = np.resize(self.excess_balanced, self.capacity_excess)
            self.excess_ids = np.resize(self.excess_ids, self.capacity_excess)

        self.starts_excess[self.size_excess] = excess_start
        self.energy_excess[self.size_excess] = excess_content
        self.excess_balanced[self.size_excess] = excess_balanced

        self.excess_ids[self.size_excess] = excess_id
        self.size_excess += 1

    def append_deficit(self, new_start, energy_remaining, balanced):
        if self.size_deficit >= self.capacity_deficit:
            self.capacity_deficit *= 2
            self.starts_deficit = np.resize(self.starts_deficit, self.capacity_deficit)
            self.energy_deficit = np.resize(self.energy_deficit, self.capacity_deficit)
            self.deficit_balanced = np.resize(self.deficit_balanced, self.capacity_deficit)

        self.starts_deficit[self.size_deficit] = new_start
        self.energy_deficit[self.size_deficit] = energy_remaining
        self.deficit_balanced[self.size_deficit] = balanced
        self.size_deficit += 1

    def remove_excess(self, index_to_remove, invalid_value=0):

        if index_to_remove < 0:
            index_to_remove += self.size_excess

        if not (0 <= index_to_remove < self.size_excess):
            raise IndexError(f"Index {index_to_remove} out of valid range (0 to {self.size_excess - 1}).")

        if index_to_remove < self.size_excess - 1:
            self.energy_excess[index_to_remove:-1] = self.energy_excess[index_to_remove + 1: self.size_excess]
            self.starts_excess[index_to_remove:-1] = self.starts_excess[index_to_remove + 1: self.size_excess]
            self.excess_balanced[index_to_remove:-1] = self.excess_balanced[index_to_remove + 1: self.size_excess]
            self.excess_ids[index_to_remove:-1] = self.excess_ids[index_to_remove + 1: self.size_excess]

        self.energy_excess[self.size_excess - 1] = invalid_value
        self.starts_excess[self.size_excess - 1] = invalid_value
        self.excess_balanced[self.size_excess - 1] = invalid_value

        self.excess_ids[self.size_excess - 1] = invalid_value

        self.size_excess -= 1

    def __eq__(self, other):
        if self.size_excess != len(other.starts_excess):
            return False

        if self.size_deficit != len(other.starts_deficit):
            return False

        return (
                np.array_equal(self.starts_excess[:self.size_excess], other.starts_excess) and
                np.array_equal(self.starts_deficit[:self.size_deficit], other.starts_deficit) and
                np.array_equal(self.energy_excess[:self.size_excess], other.energy_excess) and
                np.array_equal(self.energy_deficit[:self.size_deficit], other.energy_deficit) and
                np.array_equal(self.excess_balanced[:self.size_excess], other.excess_balanced) and
                np.array_equal(self.deficit_balanced[:self.size_deficit], other.deficit_balanced) and
                np.array_equal(self.excess_ids[:self.size_excess], other.excess_ids)
        )

    def __str__(self):
        s = f'Phase {self.id}:\n'

        filled_excess = self.starts_excess[:self.size_excess].__str__()
        filled_energy_excess = self.energy_excess[:self.size_excess].__str__()
        filled_excess_balanced = self.excess_balanced[:self.size_excess].__str__()
        filled_excess_ids = self.excess_ids[:self.size_excess].__str__()

        filled_deficit = self.starts_deficit[:self.size_deficit].__str__()
        filled_energy_deficit = self.energy_deficit[:self.size_deficit].__str__()
        filled_deficit_balanced = self.deficit_balanced[:self.size_deficit].__str__()

        s += f'starts_excess={filled_excess}, energy_excess={filled_energy_excess}, '
        s += f'excess_balanced={filled_excess_balanced}, excess_ids={filled_excess_ids}\n'

        s += f'starts_deficit={filled_deficit}, energy_deficit={filled_energy_deficit}, '
        s += f'deficit_balanced={filled_deficit_balanced}\n'

        return s

    def __repr__(self):
        return self.__str__()