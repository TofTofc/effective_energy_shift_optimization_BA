import numba
import numpy as np
from numba import njit
from numba.typed import List

GROWTH_FACTOR = 5

@njit(nogil=True, inline="always")
def resize_array(arr, new_capacity):

    rows = arr.shape[0]
    old_capacity = arr.shape[1]
    new_shape = (rows, new_capacity)
    new_arr = np.empty(new_shape, dtype=arr.dtype)
    new_arr[:, :old_capacity] = arr

    return new_arr


@njit(nogil=True, inline="always")
def add_excess_value(row_idx, start_value, energy_value,
                     starts_excess, energy_excess,
                     size_excess):

    current_size = size_excess[row_idx]
    old_capacity = starts_excess.shape[1]

    if current_size >= old_capacity:

        new_capacity = old_capacity + GROWTH_FACTOR

        starts_excess = resize_array(starts_excess, new_capacity)
        energy_excess = resize_array(energy_excess, new_capacity)

    starts_excess[row_idx, current_size] = start_value
    energy_excess[row_idx, current_size] = energy_value

    size_excess[row_idx] += 1

    return starts_excess, energy_excess


@njit(nogil=True, inline="always")
def add_deficit_value(row_idx, start_value, energy_value,
                      starts_deficit, energy_deficit, size_deficit):

    current_size = size_deficit[row_idx]
    old_capacity = starts_deficit.shape[1]

    if current_size >= old_capacity:

        new_capacity = old_capacity + GROWTH_FACTOR
        starts_deficit = resize_array(starts_deficit, new_capacity)
        energy_deficit = resize_array(energy_deficit, new_capacity)

    starts_deficit[row_idx, current_size] = start_value
    energy_deficit[row_idx, current_size] = energy_value

    size_deficit[row_idx] += 1

    return starts_deficit, energy_deficit

@njit(nogil=True, inline="always")
def insert_excess_value(row_idx, insert_idx, start_value, energy_value,
                        starts_excess, energy_excess, size_excess):

    current_size = size_excess[row_idx]
    old_capacity = starts_excess.shape[1]

    if current_size >= old_capacity:

        new_capacity = old_capacity + GROWTH_FACTOR
        starts_excess = resize_array(starts_excess, new_capacity)
        energy_excess = resize_array(energy_excess, new_capacity)

    starts_excess[row_idx, insert_idx + 1: current_size + 1] = starts_excess[row_idx, insert_idx: current_size].copy()
    energy_excess[row_idx, insert_idx + 1: current_size + 1] = energy_excess[row_idx, insert_idx: current_size].copy()

    starts_excess[row_idx, insert_idx] = start_value
    energy_excess[row_idx, insert_idx] = energy_value

    size_excess[row_idx] += 1

    return starts_excess, energy_excess