import numba
import numpy as np
from numba import njit

from versions.deficit_based.resize import add_excess_value, add_deficit_value, insert_excess_value

"""

Huge Problem: get_next_deficit_index and find_previous_deficit lead to non linear behaviour 

init capacity of 2 and growth of + 5 per resize (same as old version)
"""


@njit(nogil=True, inline="always", fastmath=True)
def get_next_deficit_index(idx, phase_meta):
    """
    Returns the idx of the next phase with deficit overflow
    """
    n = phase_meta.shape[0]
    i = (idx + 1) % n

    while True:
        if phase_meta[i, 2] == 0 and phase_meta[i, 3] == 1:
            return i
        i = (i + 1) % n


@njit(nogil=True, inline="always", fastmath=True)
def find_previous_deficit(idx, phase_meta, excess_buffer):
    count = 0
    n = phase_meta.shape[0]
    i = (idx - 1) % n

    while True:

        if phase_meta[i, 2] == 1 and phase_meta[i, 3] == 0:
            excess_buffer[count] = i
            count += 1

        if phase_meta[i, 2] == 0 and phase_meta[i, 3] == 1:
            return count

        i = (i - 1) % n


@njit(nogil=True, inline="always", fastmath=True)
def get_next_non_balanced_phase(idx, phase_meta):
    """
     Returns the idx of the next phase wich is not balanced
    """
    n = phase_meta.shape[0]
    i = (idx + 1) % n

    while True:
        if phase_meta[i, 2] == 1 or phase_meta[i, 3] == 1:
            return i
        i = (i + 1) % n


@njit(nogil=True, inline="always", fastmath=True)
def cover_deficit(current_phase_idx, excess_list,
                  phase_meta, max_height_array, e_counter, d_counter,
                  data_excess, data_deficit):
    """

    """

    for e_idx in excess_list:

        last_e_idx_src = phase_meta[e_idx, 0] - 1
        last_e_idx_dst = phase_meta[current_phase_idx, 0] - 1

        last_d_idx = phase_meta[current_phase_idx, 1] - 1

        # Get the max height inbetween the two phases
        if current_phase_idx > e_idx + 1:
            max_height = np.amax(max_height_array[e_idx + 1: current_phase_idx])

        elif current_phase_idx <= e_idx:

            current_start = e_idx + 1
            next_stop = current_phase_idx

            slice_1 = max_height_array[current_start:]

            slice_2 = max_height_array[:next_stop]

            combined_array = np.concatenate((slice_1, slice_2))

            if len(combined_array) > 0:
                max_height = np.amax(combined_array)
            else:
                max_height = 0

        else:
            max_height = 0

        # Get the excess content from the current excess
        overflow_content = data_excess[e_idx, last_e_idx_src, 1]

        # Max of the current start height and the max height of all skipped Phases
        overflow_start = np.maximum(data_excess[e_idx, last_e_idx_src, 0], max_height)

        last_excess_end_height = data_excess[current_phase_idx, last_e_idx_dst, 0] + data_excess[current_phase_idx, last_e_idx_dst, 1]

        # computed start for the moved packet (before appending)
        excess_start = np.maximum(overflow_start, last_excess_end_height)

        # merge condition:
        # start of moved packet equals end of last excess in current deficit phase
        if excess_start == last_excess_end_height:

            # Merge: increase energy of last excess in current deficit phase
            data_excess[current_phase_idx, last_e_idx_dst, 1] += overflow_content

        else:

            # rais the start of the uncovered deficit phase to match the incoming start height
            data_deficit[current_phase_idx, last_d_idx, 0] = excess_start

            # append new excess
            data_excess, phase_meta = add_excess_value(current_phase_idx, excess_start, overflow_content, data_excess, phase_meta)

        # One less excess in the excess phase
        tmp = phase_meta[e_idx, 0]
        phase_meta[e_idx, 0] = tmp - 1

        # Excess phase is now balanced
        phase_meta[e_idx, 2] = 0
        phase_meta[e_idx, 3] = 0

        # Change the max_height_array entry
        last_idx = phase_meta[e_idx, 0] - 1
        max_height_array[e_idx] = data_excess[e_idx, last_idx, 0] + data_excess[e_idx, last_idx, 1]

        e_counter -= 1

        # Check if we need more excess to cover our deficit
        def_energy = data_deficit[current_phase_idx, last_d_idx, 1]

        # Case 1: Moved Excess is smaller than deficit -> split deficit
        if overflow_content < def_energy:

            new_def_start = excess_start + overflow_content
            rem_def_energy = def_energy - overflow_content

            data_deficit[current_phase_idx, last_d_idx, 1] = overflow_content
            data_deficit, phase_meta = add_deficit_value(current_phase_idx, new_def_start, rem_def_energy, data_deficit, phase_meta)

        # Case 2: Moved Excess is larger than deficit -> split excess
        elif overflow_content > def_energy:

            new_ex_start = excess_start + def_energy
            rem_ex_energy = overflow_content - def_energy

            curr_e_idx = phase_meta[current_phase_idx, 0] - 1

            data_excess[current_phase_idx, curr_e_idx, 1] -= rem_ex_energy

            data_excess, phase_meta = insert_excess_value(current_phase_idx, curr_e_idx + 1, new_ex_start, rem_ex_energy, data_excess, phase_meta)

            phase_meta[current_phase_idx, 2] = 1
            phase_meta[current_phase_idx, 3] = 0
            d_counter -= 1
            e_counter += 1

            return e_counter, d_counter, data_excess, data_deficit, phase_meta

        # Case 3: Perfectly balanced
        else:

            phase_meta[current_phase_idx, 2] = 0
            phase_meta[current_phase_idx, 3] = 0
            d_counter -= 1
            max_height_array[current_phase_idx] = excess_start + overflow_content

            return e_counter, d_counter, data_excess, data_deficit, phase_meta

    return e_counter, d_counter, data_excess, data_deficit, phase_meta

@njit(parallel=True, nogil=True, inline="always", fastmath=True)
def init(excess_array, deficit_array):
    """
    Fills out the state mask:

    Also sets the correct height entry for max_height_array

    Returns the tuple: (Number of 1 in total, Number of -1 in total)

    phase_meta:
    0: size_excess
    1: size_deficit
    2: mask[0] (Boolean: Has phase Excess?)
    3: mask[1] (Boolean: Has phase Deficit?)


    data arrays:
    0: starting height
    1: energy excess or deficit

    """

    n = excess_array.shape[0]
    # Not smaller than 2
    initial_capacity = 2

    # float32 possible only for avg case
    data_excess = np.empty((n, initial_capacity, 2), dtype=np.float64)
    data_deficit = np.empty((n, initial_capacity, 2), dtype=np.float64)

    phase_meta = np.zeros((n, 4), dtype=np.uint8)
    max_height_array = np.zeros(n, dtype=np.float64)
    phase_meta[:, 0] = 1
    phase_meta[:, 1] = 1

    e_counter = 0
    d_counter = 0

    for i in numba.prange(n):
        data_excess[i, 0, 0] = 0
        data_excess[i, 0, 1] = excess_array[i]
        data_deficit[i, 0, 0] = 0
        data_deficit[i, 0, 1] = deficit_array[i]

        e_ex = excess_array[i]
        e_def = deficit_array[i]

        if e_ex > e_def:

            e_counter += 1
            phase_meta[i, 2] = 1
            data_excess[i, 0, 1] = e_def
            data_excess[i, 1, 0] = e_def
            data_excess[i, 1, 1] = e_ex - e_def
            phase_meta[i, 0] = 2

        elif e_def > e_ex:

            d_counter += 1
            phase_meta[i, 3] = 1
            data_deficit[i, 0, 1] = e_ex
            data_deficit[i, 1, 0] = e_ex
            data_deficit[i, 1, 1] = e_def - e_ex
            phase_meta[i, 1] = 2

        else:

            max_height_array[i] = data_excess[i, 0, 0] + data_excess[i, 0, 1]

    return e_counter, d_counter, phase_meta, max_height_array, data_excess, data_deficit


@njit(nogil=True, fastmath=True)
def process_phases(excess_array, deficit_array, start_times):

    # Provides the initial states for each Phase object and balances them
    e_counter, d_counter, phase_meta, max_height_array, data_excess, data_deficit = init(excess_array, deficit_array)

    # Return when we either start with no Excess or no Deficit
    if e_counter == 0 or d_counter == 0:
        return (
            phase_meta[:, 0], phase_meta[:, 1],
            data_excess[:, :, 0], data_deficit[:, :, 0],
            data_excess[:, :, 1], data_deficit[:, :, 1],
            phase_meta[:, 2:].T
        )

    idx = -1
    excess_inbetween_count = 0
    excess_buffer = np.empty(phase_meta.shape[0], dtype=np.int32)

    while True:

        has_excess_inbetween = False

        # Stop when either no more Excesses to move or no more Deficits to fill
        if e_counter == 0 or d_counter == 0:
            break

        while not has_excess_inbetween:

            idx = get_next_deficit_index(idx, phase_meta)
            excess_inbetween_count = find_previous_deficit(idx, phase_meta, excess_buffer)

            if excess_inbetween_count > 0:
                has_excess_inbetween = True

            else:
                idx = (idx + 1) % phase_meta.shape[0]

        # Covers the current deficit overflow phase
        e_counter, d_counter, data_excess, data_deficit, phase_meta = (cover_deficit
            (
            idx, excess_buffer[:excess_inbetween_count], phase_meta, max_height_array, e_counter, d_counter, data_excess, data_deficit
        ))

        # Stop when either no more Excesses to move or no more Deficits to fill
        if e_counter == 0 or d_counter == 0:
            break

        # Index +1
        idx = (idx + 1) % phase_meta.shape[0]

    return \
        (
            phase_meta[:, 0], phase_meta[:, 1],
            data_excess[:, :, 0], data_deficit[:, :, 0],
            data_excess[:, :, 1], data_deficit[:, :, 1],
            phase_meta[:, 2:].T
        )