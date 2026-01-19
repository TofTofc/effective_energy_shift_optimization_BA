import numba
import numpy as np
from numba import njit

from versions.new_version_fusion_phaseless_2_array_structure_no_max_height.resize import add_excess_value, add_deficit_value, insert_excess_value

@njit(nogil = True, inline = "always", fastmath = True)
def get_next_excess_index(idx, phase_meta):
    """
    Returns the idx of the next phase with excess overflow
    """
    n = phase_meta.shape[0]
    i = (idx + 1) % n

    while True:
        if phase_meta[i, 3] == 1 and phase_meta[i, 4] == 0:
            return i
        i = (i + 1) % n


@njit(nogil = True, inline = "always", fastmath = True)
def get_next_non_balanced_phase(idx, phase_meta):

    """
     Returns the idx of the next phase wich is not balanced
    """
    n = phase_meta.shape[0]
    i = (idx + 1) % n

    while True:
        if phase_meta[i, 3] == 1 or phase_meta[i, 4] == 1:
            return i
        i = (i + 1) % n

@njit(nogil = True, inline = "always", fastmath = True)
def move_excess(current_phase_idx, next_phase_idx,
                phase_meta, e_counter, d_counter,
                data_excess, data_deficit):

    """
    Moves all uncovered excess packets from current -> next phase

    1. Compute max height of skipped phases (0 if no phases skipped)
    2. Take all uncovered excess packets from current phase
    3. For each of those packets: new start = max(orig_start, max_skipped_height, end_of_last_in_next)
    4. Insert packet into next phase via append_excess
    4.1 merge the new excess in the next phase if possible
    5. Update uncovered excess counters in both phases
    6. Remove transferred packets from current phase
    7. Mark current phase as balanced: mask=0, e_counter--
    8. Handle new packets in next phase by calling balance_phase
    Returns updated (e_counter, d_counter).
    """

    # Get start index of not covered excesses
    n_not_cov = phase_meta[current_phase_idx, 2]
    total = phase_meta[current_phase_idx, 0]
    start_idx = total - n_not_cov

    # Iterates over all uncovered excesses
    for idx in range(start_idx, total):

        # Get the excess content from the current uncovered excess
        overflow_content = data_excess[current_phase_idx, idx, 1]

        # current start height
        overflow_start = data_excess[current_phase_idx, idx, 0]

        last_idx_next = phase_meta[next_phase_idx, 0] - 1
        last_excess_end_height = data_excess[next_phase_idx, last_idx_next, 0] + data_excess[next_phase_idx, last_idx_next, 1]

        # computed start for the moved packet (before appending)
        excess_start = np.maximum(overflow_start, last_excess_end_height)

        # merge conditions:
        # 1. there is at least one uncovered excess in next phase
        # 2. start of moved packet equals end of last excess in next phase
        can_merge = (phase_meta[next_phase_idx, 2] > 0) and (abs(excess_start - last_excess_end_height) < 1e-12)

        if can_merge:

            # Merge: increase energy of last excess in next_phase
            data_excess[next_phase_idx, last_idx_next, 1] += overflow_content
        # not increment next_phase.number_of_excess_not_covered because we no new packet

        else:

            # append new excess
            #i = size_excess[next_phase_idx]
            #starts_excess[next_phase_idx, i] = excess_start
            #energy_excess[next_phase_idx, i] = overflow_content
            #size_excess[next_phase_idx] += 1
            phase_meta[next_phase_idx, 2] += 1
            data_excess, phase_meta = add_excess_value(next_phase_idx, excess_start, overflow_content, data_excess, phase_meta)

        phase_meta[current_phase_idx, 2] -= 1

    phase_meta[current_phase_idx, 0] = start_idx

    # Current phase is now balanced
    phase_meta[current_phase_idx, 3] = 0
    phase_meta[current_phase_idx, 4] = 0

    e_counter -= 1

    e_counter, d_counter, data_excess, data_deficit, phase_meta = (
        balance_phase(next_phase_idx, phase_meta, e_counter, d_counter, data_excess, data_deficit)
    )

    return e_counter, d_counter, data_excess, data_deficit, phase_meta


@njit(nogil = True, inline = "always", fastmath = True)
def balance_phase(i, phase_meta, e_counter, d_counter, data_excess, data_deficit):

    """
    Balances newly moved excess packets for the phase

    Behaviour:
    0. If there is no uncovered deficit block, nothing to do for balancing (number_of_excess_not_covered was already changed properly)

    1. Start at the first not covered excess and iterate over all of them
    2. Raise the start of the last deficit packet (uncovered one) to the start of the current excess
    3. For each current excess vs the uncovered deficit one of 3 happens:
        a: excess < deficit:
            - Split the deficit: the lower part is matched to the excess (becomes covered)
              the upper part remains as a smaller deficit starting at the end of the matched pair
            - number_of_excess_not_covered --
            - Move to the next excess and continue
       b: excess > deficit:
            - Split the excess: the lower part covers the deficit, the remaining excess stays
              (with its start adjusted to the end of the covered portion)
            - Update: deficit counter --, excess counter++, state_mask[i] = 1
            - return
       c: excess == deficit:
            - If last uncovered excess: state_mask[i] = 0, deficit counter --
              number_of_excess_not_covered --
            - If not last excess: state_mask[i] = 1, deficit counter--, excess counter++
              number_of_excess_not_covered -- and return

    4. Return updated (e_counter, d_counter)
    """

    # 0. no uncovered deficit block -> nothing to do
    if phase_meta[i, 4] == 0:

        return e_counter, d_counter, data_excess, data_deficit, phase_meta

    #  1. Start at the first not covered excess and iterate over all of them
    n_not_cov = phase_meta[i, 2]
    total = phase_meta[i, 0]
    start_idx = total - n_not_cov

    for idx in range(start_idx, total):

        # 2. Raise the start of the last deficit packet (uncovered one) to the start of the current excess

        last_def_idx = phase_meta[i, 1] - 1
        data_deficit[i, last_def_idx, 0] = data_excess[i, idx, 0]

        excess_energy = data_excess[i, idx, 1]
        deficit_energy = data_deficit[i, last_def_idx, 1]
        deficit_start = data_deficit[i, last_def_idx, 0]

        # 3. For each current excess vs the uncovered deficit one of 3 happens:

        diff = excess_energy - deficit_energy

        # a: excess < deficit:
        if diff < -1e-12:

            # Split the deficit

            # New start is excess height + start
            new_start = data_excess[i, idx, 0] + excess_energy

            # Remaining deficit is current deficit - energy excess
            energy_remaining = deficit_energy - excess_energy

            data_deficit, phase_meta = add_deficit_value(
                i, new_start, energy_remaining, data_deficit, phase_meta
            )

            # Change Deficit of lower packet
            data_deficit[i, last_def_idx, 1] = excess_energy

            # number_of_excess_not_covered--
            phase_meta[i, 2] -= 1

            # Move to the next excess and continue
            continue

        # b: excess > deficit
        elif diff > 1e-12:

            # computed start for the remaining excess
            new_start = deficit_start + deficit_energy
            energy_remaining = excess_energy - deficit_energy

            # set lower packet to cover the deficit
            data_excess[i, idx, 1] = deficit_energy

            # insert remaining excess after the covered excess and NOT at the end to keep correct sequence
            # phase.append_excess(new_start, energy_remaining, phase.get_excess_id(idx))
            insert_idx = idx + 1
            data_excess, phase_meta = insert_excess_value(
                i, insert_idx, new_start, energy_remaining, data_excess, phase_meta
            )

            # update counters and mark phase as still having excess
            d_counter -= 1
            e_counter += 1
            phase_meta[i, 3] = 1
            phase_meta[i, 4] = 0

            # return updated counters
            return e_counter, d_counter, data_excess, data_deficit, phase_meta

        # c: excess == deficit:
        else:

            # If last (uncovered) excess
            if idx == total-1:

                #state_mask[i] = 0, deficit counter --
                phase_meta[i, 3] = 0
                phase_meta[i, 4] = 0
                d_counter -= 1

                # number_of_excess_not_covered --
                phase_meta[i, 2] -= 1

                return e_counter, d_counter, data_excess, data_deficit, phase_meta

            # If not last (uncovered) excess
            else:

                # state_mask[i] = 1, deficit counter--, excess counter++
                phase_meta[i, 3] = 1
                phase_meta[i, 4] = 0
                d_counter -= 1
                e_counter += 1

                # number_of_excess_not_covered --
                phase_meta[i, 2] -= 1

                return e_counter, d_counter, data_excess, data_deficit, phase_meta

    return e_counter, d_counter, data_excess, data_deficit, phase_meta

@njit(parallel = True, nogil = True, inline = "always", fastmath = True)
def init(excess_array, deficit_array):
    """
    Fills out the state mask:

    Returns the tuple: (Number of 1 in total, Number of -1 in total)

    phase_meta:
    0: size_excess
    1: size_deficit
    2: number_of_excess_not_covered
    3: mask[0] (Boolean: Has phase Excess?)
    4: mask[1] (Boolean: Has phase Deficit?)


    data arrays:
    0: starting height
    1: energy excess or deficit

    """

    n = excess_array.shape[0]
    # Not smaller than 2
    initial_capacity = 2

    data_excess = np.empty((n, initial_capacity, 2), dtype=np.float64)
    data_deficit = np.empty((n, initial_capacity, 2), dtype=np.float64)

    phase_meta = np.zeros((n, 5), dtype=np.uint8)
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

        diff = e_ex - e_def

        if diff > 1e-12:

            e_counter += 1
            phase_meta[i, 3] = 1
            data_excess[i, 0, 1] = e_def
            data_excess[i, 1, 0] = e_def
            data_excess[i, 1, 1] = e_ex - e_def
            phase_meta[i, 0] = 2
            phase_meta[i, 2] = 1

        elif diff < -1e-12:

            d_counter += 1
            phase_meta[i, 4] = 1
            data_deficit[i, 0, 1] = e_ex
            data_deficit[i, 1, 0] = e_ex
            data_deficit[i, 1, 1] = e_def - e_ex
            phase_meta[i, 1] = 2


    return e_counter, d_counter, phase_meta, data_excess, data_deficit

@njit(nogil = True, fastmath = True)
def process_phases(excess_array, deficit_array, start_times):

    # Provides the initial states for each Phase object and balances them
    e_counter, d_counter, phase_meta, data_excess, data_deficit = init(excess_array, deficit_array)

    # Return when we either start with no Excess or no Deficit
    if e_counter == 0 or d_counter == 0:
        return (
            phase_meta[:, 0], phase_meta[:, 1],
            data_excess[:, :, 0], data_deficit[:, :, 0],
            data_excess[:, :, 1], data_deficit[:, :, 1],
            phase_meta[:, 3:].T
        )

    # start with an excess overflow right away
    idx = get_next_excess_index(0, phase_meta)

    while True:

        # Stop when either no more Excesses to move or no more Deficits to fill
        if e_counter == 0 or d_counter == 0:
            break

        # For each Phase there are 3 possibilities

        #1. Excess > Deficit
        next_phase_idx = get_next_non_balanced_phase(idx, phase_meta)

        # Moves the Excess from the current Phase to the next non perfectly balanced phase
        e_counter, d_counter, data_excess, data_deficit, phase_meta = move_excess(
            idx, next_phase_idx, phase_meta, e_counter, d_counter, data_excess, data_deficit
        )

        # Stop when either no more Excesses to move or no more Deficits to fill
        if e_counter == 0 or d_counter == 0:
            break

        #2. Excess = Deficit (cant happen)
        # Nothing to move here

        #3. Excess < Deficit (cant happen)
        # Nothing to move here

        # Index goes to the next Excess
        idx = get_next_excess_index(idx, phase_meta)

    return \
    (
        phase_meta[:, 0], phase_meta[:, 1],
        data_excess[:, :, 0], data_deficit[:, :, 0],
        data_excess[:, :, 1], data_deficit[:, :, 1],
        phase_meta[:, 3:].T
    )