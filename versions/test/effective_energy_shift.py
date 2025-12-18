import numba
import numpy as np
from numba import njit
from numba.typed import List

from versions.test.resize import add_excess_value, add_deficit_value, insert_excess_value

"""
changes made from new_version_fusion_2_avg_case_dtypes:

- changed % usage to if i >= n: i = 0
- grouped meta info arrays into a big array called phase_meta

init capacity of 2 and growth of + 5 per resize (same as old version)
"""

@njit(nogil = True, inline = "always")
def get_next_excess_index(idx, phase_meta):
    """
    Returns the idx of the next phase with excess overflow
    """
    n = phase_meta.shape[0]
    i = idx + 1
    if i >= n: i = 0

    while True:
        if phase_meta[i, 4] == 1 and phase_meta[i, 5] == 0:
            return i
        i += 1
        if i >= n: i = 0


@njit(nogil = True, inline = "always")
def get_next_non_balanced_phase(idx, phase_meta):

    """
     Returns the idx of the next phase wich is not balanced
    """
    n = phase_meta.shape[0]
    i = idx + 1
    if i >= n: i = 0

    while True:
        if phase_meta[i, 4] == 1 or phase_meta[i, 5] == 1:
            return i
        i += 1
        if i >= n: i = 0

@njit(nogil = True, inline = "always")
def move_excess(current_phase_idx, next_phase_idx,
                phase_meta,
                e_counter, d_counter,
                starts_excess, energy_excess,
                starts_deficit, energy_deficit):

    """
    Moves all uncovered excess packets from current -> next phase

    1. Compute max height of skipped phases (0 if no phases skipped)
    2. Take all uncovered excess packets from current phase
    3. For each of those packets: new start = max(orig_start, max_skipped_height, end_of_last_in_next)
    4. Insert packet into next phase via append_excess
    4.1 merge the new excess in the next phase if possible
    5. Update uncovered excess counters in both phases
    6. Remove transferred packets from current phase
    7. Mark current phase as balanced: mask=0, e_counter--, update max_height
    8. Handle new packets in next phase by calling balance_phase
    Returns updated (e_counter, d_counter).
    """

    # Get the max height inbetween the two phases
    if next_phase_idx > current_phase_idx + 1:
        max_height = np.amax(phase_meta[current_phase_idx + 1: next_phase_idx, 3])

    elif next_phase_idx <= current_phase_idx:

        current_start = current_phase_idx + 1
        next_stop = next_phase_idx

        slice_1 = phase_meta[current_start:, 3]

        slice_2 = phase_meta[:next_stop, 3]

        combined_array = np.concatenate((slice_1, slice_2))

        if len(combined_array) > 0:
            max_height = np.amax(combined_array)
        else:
            max_height = 0

    else:
        max_height = 0

    # Get start index of not covered excesses
    n_not_cov = phase_meta[current_phase_idx, 2]
    total = phase_meta[current_phase_idx, 0]
    start_idx = total - n_not_cov

    # Iterates over all uncovered excesses
    for idx in range(start_idx, total):

        # Get the excess content from the current uncovered excess
        overflow_content = energy_excess[current_phase_idx, idx]

        # Max of the current start height and the max height of all skipped Phases
        overflow_start = np.maximum(starts_excess[current_phase_idx, idx], max_height)

        last_idx_next = phase_meta[next_phase_idx, 0] - 1
        last_excess_end_height = starts_excess[next_phase_idx, last_idx_next] + energy_excess[next_phase_idx, last_idx_next]

        # computed start for the moved packet (before appending)
        excess_start = np.maximum(overflow_start, last_excess_end_height)

        # merge conditions:
        # 1. there is at least one uncovered excess in next phase
        # 2. start of moved packet equals end of last excess in next phase
        can_merge = (phase_meta[next_phase_idx, 2] > 0) and (excess_start == last_excess_end_height)

        if can_merge:

            # Merge: increase energy of last excess in next_phase
            energy_excess[next_phase_idx, last_idx_next] += overflow_content
            # not increment next_phase.number_of_excess_not_covered because we no new packet
        else:

            # append new excess
            #i = size_excess[next_phase_idx]
            #starts_excess[next_phase_idx, i] = excess_start
            #energy_excess[next_phase_idx, i] = overflow_content
            #size_excess[next_phase_idx] += 1
            phase_meta[next_phase_idx, 2] += 1

            starts_excess, energy_excess, phase_meta = add_excess_value(
                row_idx=next_phase_idx,
                start_value=excess_start,
                energy_value=overflow_content,
                starts_excess=starts_excess,
                energy_excess=energy_excess,
                phase_meta=phase_meta
            )

        # remove one uncovered excess from current phase
        phase_meta[current_phase_idx, 2] -= 1

    phase_meta[current_phase_idx, 0] = start_idx

    # Current phase is now balanced
    phase_meta[current_phase_idx, 4] = 0
    phase_meta[current_phase_idx, 5] = 0

    # Change the max_height_array entry

    last_idx = phase_meta[current_phase_idx, 0] - 1
    phase_meta[current_phase_idx, 3] = starts_excess[current_phase_idx, last_idx] + energy_excess[current_phase_idx, last_idx]

    e_counter -= 1

    e_counter, d_counter, starts_excess, energy_excess, starts_deficit, energy_deficit, phase_meta = (
        balance_phase(next_phase_idx, phase_meta, e_counter, d_counter,
                starts_excess, energy_excess,
                starts_deficit, energy_deficit))

    return e_counter, d_counter, starts_excess, energy_excess, starts_deficit, energy_deficit, phase_meta


@njit(nogil = True, inline = "always")
def balance_phase(i, phase_meta, e_counter, d_counter,
                  starts_excess, energy_excess,
                  starts_deficit, energy_deficit,):
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
              set max_height_array[i] to the end height of this block and return
              number_of_excess_not_covered --
            - If not last excess: state_mask[i] = 1, deficit counter--, excess counter++
              number_of_excess_not_covered -- and return

    4. Return updated (e_counter, d_counter)
    """

    # 0. no uncovered deficit block -> nothing to do
    if phase_meta[i, 5] == 0:

        return e_counter, d_counter, starts_excess, energy_excess, starts_deficit, energy_deficit, phase_meta

    #  1. Start at the first not covered excess and iterate over all of them
    n_not_cov = phase_meta[i, 2]
    total = phase_meta[i, 0]
    start_idx = total - n_not_cov

    for idx in range(start_idx, total):

        # 2. Raise the start of the last deficit packet (uncovered one) to the start of the current excess

        last_def_idx = phase_meta[i, 1] - 1
        starts_deficit[i, last_def_idx] = starts_excess[i, idx]

        excess_energy = energy_excess[i, idx]
        deficit_energy = energy_deficit[i, last_def_idx]
        deficit_start = starts_deficit[i, last_def_idx]

        # 3. For each current excess vs the uncovered deficit one of 3 happens:

        # a: excess < deficit:
        if excess_energy < deficit_energy:
            # Split the deficit

            # New start is excess height + start
            new_start = starts_excess[i, idx] + excess_energy

            # Remaining deficit is current deficit - energy excess
            energy_remaining = deficit_energy  - excess_energy

            #next_def_idx = size_deficit[i]
            #starts_deficit[i, next_def_idx] = new_start
            #energy_deficit[i, next_def_idx] = energy_remaining
            #size_deficit[i] += 1

            starts_deficit, energy_deficit, phase_meta = add_deficit_value(
                row_idx=i,
                start_value=new_start,
                energy_value=energy_remaining,
                starts_deficit=starts_deficit,
                energy_deficit=energy_deficit,
                phase_meta=phase_meta,
            )

            # Change Deficit of lower packet
            energy_deficit[i, last_def_idx] = excess_energy

            # number_of_excess_not_covered--
            phase_meta[i, 2] -= 1

            # Move to the next excess and continue
            continue

        # b: excess > deficit
        elif excess_energy > deficit_energy:

            # computed start for the remaining excess
            new_start = deficit_start + deficit_energy
            energy_remaining = excess_energy - deficit_energy

            # set lower packet to cover the deficit
            energy_excess[i, idx] = deficit_energy

            # insert remaining excess after the covered excess and NOT at the end to keep correct sequence
            # phase.append_excess(new_start, energy_remaining, phase.get_excess_id(idx))
            insert_idx = idx + 1

            #size = size_excess[i]

            #starts_excess[i, insert_idx + 1: size + 1] = starts_excess[i, insert_idx:size].copy()
            #energy_excess[i, insert_idx + 1: size + 1] = energy_excess[i, insert_idx:size].copy()

            #starts_excess[i, insert_idx] = new_start
            #energy_excess[i, insert_idx] = energy_remaining
            #size_excess[i] += 1

            starts_excess, energy_excess, phase_meta = insert_excess_value(
                row_idx=i,
                insert_idx=insert_idx,
                start_value=new_start,
                energy_value=energy_remaining,
                starts_excess=starts_excess,
                energy_excess=energy_excess,
                phase_meta=phase_meta,
            )

            # update counters and mark phase as still having excess
            d_counter -= 1
            e_counter += 1
            phase_meta[i, 4] = 1
            phase_meta[i, 5] = 0

            # return updated counters
            return e_counter, d_counter, starts_excess, energy_excess, starts_deficit, energy_deficit, phase_meta

        # c: excess == deficit:
        else:

            # If last (uncovered) excess
            if idx == total-1:

                #state_mask[i] = 0, deficit counter --
                phase_meta[i, 4] = 0
                phase_meta[i, 5] = 0
                d_counter -= 1

                # set max_height_array[i] to the end height of this block
                phase_meta[i, 3] = starts_excess[i, idx] + energy_excess[i, idx]

                # number_of_excess_not_covered --
                phase_meta[i, 2] -= 1

                return e_counter, d_counter, starts_excess, energy_excess, starts_deficit, energy_deficit, phase_meta

            # If not last (uncovered) excess
            else:

                # state_mask[i] = 1, deficit counter--, excess counter++
                phase_meta[i, 4] = 1
                phase_meta[i, 5] = 0
                d_counter -= 1
                e_counter += 1

                # number_of_excess_not_covered --
                phase_meta[i, 2] -= 1

                return e_counter, d_counter, starts_excess, energy_excess, starts_deficit, energy_deficit, phase_meta

    return e_counter, d_counter, starts_excess, energy_excess, starts_deficit, energy_deficit, phase_meta

@njit(parallel = True, nogil = True, inline = "always")
def init(excess_array, deficit_array):
    """
    Fills out the state mask:

    Also sets the correct height entry for max_height_array

    Returns the tuple: (Number of 1 in total, Number of -1 in total)

    phase_meta:
    0: size_excess
    1: size_deficit
    2: number_of_excess_not_covered
    3: max_height_array
    4: mask[0] (Boolean: Has phase Excess?)
    5: mask[1] (Boolean: Has phase Deficit?)

    """

    n = excess_array.shape[0]
    # Not smaller than 2
    initial_capacity = 2

    # Use uint64 for worst case
    starts_excess = np.empty((n, initial_capacity), dtype=np.uint32)
    starts_deficit = np.empty((n, initial_capacity), dtype=np.uint32)
    energy_excess = np.empty((n, initial_capacity), dtype=np.uint32)

    # Uint8 due to max of 100 deficit in our case needs to be higher if input ints can be higher than 255
    energy_deficit = np.empty((n, initial_capacity), dtype=np.uint8)

    phase_meta = np.zeros((n, 6), dtype=np.uint32)
    phase_meta[:, 0] = 1
    phase_meta[:, 1] = 1

    e_counter = 0
    d_counter = 0

    for i in numba.prange(n):

        starts_excess[i, 0] = 0
        starts_deficit[i, 0] = 0
        energy_excess[i, 0] = excess_array[i]
        energy_deficit[i, 0] = deficit_array[i]

        e_ex = excess_array[i]
        e_def = deficit_array[i]

        if e_ex > e_def:

            e_counter += 1
            phase_meta[i, 4] = 1

            energy_excess[i, 0] = e_def

            energy_excess[i, 1] = e_ex - e_def
            starts_excess[i, 1] = e_def

            phase_meta[i, 0] = 2
            phase_meta[i, 2] = 1

        elif e_def > e_ex:

            d_counter += 1
            phase_meta[i, 5] = 1

            energy_deficit[i, 0] = e_ex

            energy_deficit[i, 1] = e_def - e_ex
            starts_deficit[i, 1] = e_ex

            phase_meta[i, 1] = 2

        else:

            phase_meta[i, 3] = (starts_excess[i, 0] + energy_excess[i, 0])

    return (e_counter, d_counter, phase_meta,
            starts_excess, starts_deficit,
            energy_excess, energy_deficit)

@njit(nogil = True)
def process_phases(excess_array, deficit_array, start_times):

    # Provides the initial states for each Phase object and balances them
    (e_counter, d_counter, phase_meta,
     starts_excess, starts_deficit,
     energy_excess, energy_deficit
     ) = init(excess_array, deficit_array)

    # Return when we either start with no Excess or no Deficit
    if e_counter == 0 or d_counter == 0:
        return \
            (
                phase_meta[:, 0], phase_meta[:, 1],
                starts_excess, starts_deficit,
                energy_excess, energy_deficit,
                phase_meta[:, 4:].T
            )

    # start with an excess overflow right away
    idx = get_next_excess_index( 0, phase_meta)

    while True:

        # Stop when either no more Excesses to move or no more Deficits to fill
        if e_counter == 0 or d_counter == 0:
            break

        # For each Phase there are 3 possibilities

        #1. Excess > Deficit
        next_phase_idx = get_next_non_balanced_phase(idx, phase_meta)

        # Moves the Excess from the current Phase to the next non perfectly balanced phase
        e_counter, d_counter, starts_excess, energy_excess, starts_deficit, energy_deficit, phase_meta = move_excess(
            idx, next_phase_idx,
            phase_meta,
            e_counter, d_counter,
            starts_excess, energy_excess,
            starts_deficit, energy_deficit
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
        starts_excess, starts_deficit,
        energy_excess, energy_deficit,
        phase_meta[:, 4:].T
    )