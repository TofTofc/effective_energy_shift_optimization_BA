import numpy as np
from numba import njit

GROWTH_FACTOR = 5

@njit(nogil=True, inline="always")
def resize_array(arr, new_capacity):
    rows = arr.shape[0]
    old_capacity = arr.shape[1]
    new_shape = (rows, new_capacity, 2)
    # Nutze zeros statt empty, um e-312 (Müllwerte) in neuen Slots zu verhindern
    new_arr = np.zeros(new_shape, dtype=arr.dtype)
    new_arr[:, :old_capacity, :] = arr
    return new_arr

@njit(nogil=True, inline="always")
def add_excess_value(row_idx, start_value, energy_value, data_excess, phase_meta):
    current_size = int(phase_meta[row_idx, 0])
    old_capacity = data_excess.shape[1]

    if current_size >= old_capacity:
        new_capacity = old_capacity + GROWTH_FACTOR
        data_excess = resize_array(data_excess, new_capacity)

    data_excess[row_idx, current_size, 0] = start_value
    data_excess[row_idx, current_size, 1] = energy_value
    phase_meta[row_idx, 0] += 1

    return data_excess, phase_meta

@njit(nogil=True, inline="always")
def add_deficit_value(row_idx, start_value, energy_value, data_deficit, phase_meta):
    current_size = int(phase_meta[row_idx, 1])
    old_capacity = data_deficit.shape[1]

    if current_size >= old_capacity:
        new_capacity = old_capacity + GROWTH_FACTOR
        data_deficit = resize_array(data_deficit, new_capacity)

    data_deficit[row_idx, current_size, 0] = start_value
    data_deficit[row_idx, current_size, 1] = energy_value
    phase_meta[row_idx, 1] += 1

    return data_deficit, phase_meta

@njit(nogil=True, inline="always")
def insert_excess_value(row_idx, insert_idx, start_value, energy_value, data_excess, phase_meta):
    current_size = int(phase_meta[row_idx, 0])
    old_capacity = data_excess.shape[1]

    if current_size >= old_capacity:
        new_capacity = old_capacity + GROWTH_FACTOR
        data_excess = resize_array(data_excess, new_capacity)

    # Shifting mit Slicing und Copy (wie von dir gewünscht)
    if insert_idx < current_size:
        data_excess[row_idx, insert_idx + 1: current_size + 1, :] = \
            data_excess[row_idx, insert_idx: current_size, :].copy()

    data_excess[row_idx, insert_idx, 0] = start_value
    data_excess[row_idx, insert_idx, 1] = energy_value
    phase_meta[row_idx, 0] += 1

    return data_excess, phase_meta

@njit(nogil=True, inline="always")
def insert_deficit_value(row_idx, insert_idx, start_value, energy_value, data_deficit, phase_meta):
    current_size = int(phase_meta[row_idx, 1])
    old_capacity = data_deficit.shape[1]

    if current_size >= old_capacity:
        new_capacity = old_capacity + GROWTH_FACTOR
        data_deficit = resize_array(data_deficit, new_capacity)

    # Shifting mit Slicing und Copy
    if insert_idx < current_size:
        data_deficit[row_idx, insert_idx + 1: current_size + 1, :] = \
            data_deficit[row_idx, insert_idx: current_size, :].copy()

    data_deficit[row_idx, insert_idx, 0] = start_value
    data_deficit[row_idx, insert_idx, 1] = energy_value
    phase_meta[row_idx, 1] += 1

    return data_deficit, phase_meta