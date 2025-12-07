import numba
import numpy as np
from numba import njit
from numba.typed import List

@njit(nogil = True, inline = "always")
def get_next_excess_index(idx, state_mask):
    """
    Returns the idx of the next phase with excess overflow
    """
    n = state_mask.shape[1]
    i = (idx + 1) % n

    while True:
        if state_mask[0][i] and not state_mask[1][i]:
            return i
        i = (i + 1) % n


@njit(nogil = True, inline = "always")
def get_next_non_balanced_phase(idx, state_mask):

    """
     Returns the idx of the next phase wich is not balanced
    """
    n = state_mask.shape[1]
    i = (idx + 1) % n

    while True:
        if state_mask[0][i] or state_mask[1][i]:
            return i
        i = (i + 1) % n

@njit(nogil = True, inline = "always")
def move_excess(current_phase_idx, next_phase_idx,
                max_height_array, mask,
                e_counter, d_counter,
                size_excess,
                number_of_excess_not_covered,
                starts_excess, energy_excess,
                size_deficit, starts_deficit, energy_deficit):

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
        max_height = np.amax(max_height_array[current_phase_idx + 1: next_phase_idx])

    elif next_phase_idx <= current_phase_idx:

        current_start = current_phase_idx + 1
        next_stop = next_phase_idx

        slice_1 = max_height_array[current_start:]

        slice_2 = max_height_array[:next_stop]

        combined_array = np.concatenate((slice_1, slice_2))

        if len(combined_array) > 0:
            max_height = np.amax(combined_array)
        else:
            max_height = 0

    else:
        max_height = 0

    # Get start index of not covered excesses
    n = number_of_excess_not_covered[current_phase_idx]
    total = size_excess[current_phase_idx]
    start_idx = total - n

    # Iterates over all uncovered excesses
    for idx in range(start_idx, total):

        # Get the excess content from the current uncovered excess
        overflow_content = energy_excess[current_phase_idx, idx]

        # Max of the current start height and the max height of all skipped Phases
        overflow_start = max(starts_excess[current_phase_idx, idx], max_height)

        last_idx_next = size_excess[next_phase_idx] - 1
        last_excess_end_height = starts_excess[next_phase_idx, last_idx_next] + energy_excess[next_phase_idx, last_idx_next]

        # computed start for the moved packet (before appending)
        excess_start = max(overflow_start, last_excess_end_height)

        # merge conditions:
        # 1. there is at least one uncovered excess in next phase
        # 2. start of moved packet equals end of last excess in next phase
        can_merge = (number_of_excess_not_covered[next_phase_idx] > 0) and (excess_start == last_excess_end_height)

        if can_merge:

            # Merge: increase energy of last excess in next_phase
            energy_excess[next_phase_idx, last_idx_next] += overflow_content
            # not increment next_phase.number_of_excess_not_covered because we no new packet
        else:

            # append new excess
            i = size_excess[next_phase_idx]
            starts_excess[next_phase_idx, i] = excess_start
            energy_excess[next_phase_idx, i] = overflow_content
            size_excess[next_phase_idx] += 1
            number_of_excess_not_covered[next_phase_idx] += 1

        # remove one uncovered excess from current phase
        number_of_excess_not_covered[current_phase_idx] -= 1

    size_excess[current_phase_idx] = start_idx

    # Current phase is now balanced
    mask[0][current_phase_idx] = False
    mask[1][current_phase_idx] = False

    # Change the max_height_array entry

    last_idx = size_excess[current_phase_idx] - 1
    max_height_array[current_phase_idx] = starts_excess[current_phase_idx, last_idx] + energy_excess[current_phase_idx, last_idx]

    e_counter -= 1

    e_counter, d_counter = balance_phase(next_phase_idx, mask, max_height_array, e_counter, d_counter,
                  size_excess, number_of_excess_not_covered,
                  starts_excess, energy_excess,
                  size_deficit, starts_deficit, energy_deficit)

    return e_counter, d_counter

@njit(nogil = True, inline = "always")
def balance_phase(i, mask, max_height_array, e_counter, d_counter,
                  size_excess, number_of_excess_not_covered,
                  starts_excess, energy_excess,
                  size_deficit, starts_deficit, energy_deficit):
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
    if not mask[1][i]:
        return e_counter, d_counter

    #  1. Start at the first not covered excess and iterate over all of them
    n = number_of_excess_not_covered[i]
    total = size_excess[i]
    start_idx = total - n

    for idx in range(start_idx, total):

        # 2. Raise the start of the last deficit packet (uncovered one) to the start of the current excess

        last_def_idx = size_deficit[i] - 1
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

            next_def_idx = size_deficit[i]
            starts_deficit[i, next_def_idx] = new_start
            energy_deficit[i, next_def_idx] = energy_remaining
            size_deficit[i] += 1

            # Change Deficit of lower packet
            energy_deficit[i, last_def_idx] = excess_energy

            # number_of_excess_not_covered--
            number_of_excess_not_covered[i] -= 1

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
            size = size_excess[i]

            starts_excess[i, insert_idx + 1: size + 1] = starts_excess[i, insert_idx:size].copy()
            energy_excess[i, insert_idx + 1: size + 1] = energy_excess[i, insert_idx:size].copy()

            starts_excess[i, insert_idx] = new_start
            energy_excess[i, insert_idx] = energy_remaining
            size_excess[i] += 1

            # update counters and mark phase as still having excess
            d_counter -= 1
            e_counter += 1
            mask[0][i] = True
            mask[1][i] = False

            # return updated counters
            return e_counter, d_counter

        # c: excess == deficit:
        else:

            # If last (uncovered) excess
            if idx == total-1:

                #state_mask[i] = 0, deficit counter --
                mask[0][i] = False
                mask[1][i] = False
                d_counter -= 1

                # set max_height_array[i] to the end height of this block
                max_height_array[i] = starts_excess[i, idx] + energy_excess[i, idx]

                # number_of_excess_not_covered --
                number_of_excess_not_covered[i] -= 1

                return e_counter, d_counter

            # If not last (uncovered) excess
            else:

                # state_mask[i] = 1, deficit counter--, excess counter++
                mask[0][i] = True
                mask[1][i] = False
                d_counter -= 1
                e_counter += 1

                # number_of_excess_not_covered --
                number_of_excess_not_covered[i] -= 1

                return e_counter, d_counter

    return e_counter, d_counter

@njit(parallel = True, nogil = True, inline = "always")
def init(excess_array, deficit_array, start_times):
    """
    Fills out the state mask:

    Also sets the correct height entry for max_height_array

    Returns the tuple: (Number of 1 in total, Number of -1 in total)
    """

    n = excess_array.shape[0]
    initial_capacity = 50

    starts_excess = np.empty((n, initial_capacity), dtype=np.int64)
    starts_deficit = np.empty((n, initial_capacity), dtype=np.int64)
    energy_excess = np.empty((n, initial_capacity), dtype=np.int32)
    energy_deficit = np.empty((n, initial_capacity), dtype=np.int32)

    capacity_excess = np.empty(n, dtype=np.int32)
    capacity_deficit = np.empty(n, dtype=np.int32)
    size_excess = np.empty(n, dtype=np.int32)
    size_deficit = np.empty(n, dtype=np.int32)
    number_of_excess_not_covered = np.empty(n, dtype=np.int32)

    mask = np.ones((2, n), dtype=np.bool_)
    max_height_array = np.zeros(n, dtype=np.int64)

    e_counter = 0
    d_counter = 0

    for i in numba.prange(n):

        capacity_excess[i] = initial_capacity
        capacity_deficit[i] = initial_capacity

        starts_excess[i, 0] = 0
        starts_deficit[i, 0] = 0
        energy_excess[i, 0] = excess_array[i]
        energy_deficit[i, 0] = deficit_array[i]

        e_ex = excess_array[i]
        e_def = deficit_array[i]

        if e_ex > e_def:

            e_counter += 1
            mask[0, i] = True
            mask[1, i] = False

            energy_excess[i, 0] = e_def

            energy_excess[i, 1] = e_ex - e_def
            starts_excess[i, 1] = e_def

            size_excess[i] = 2
            size_deficit[i] = 1
            number_of_excess_not_covered[i] = 1

        elif e_def > e_ex:

            d_counter += 1
            mask[0, i] = False
            mask[1, i] = True

            energy_deficit[i, 0] = e_ex

            energy_deficit[i, 1] = e_def - e_ex
            starts_deficit[i, 1] = e_ex

            size_excess[i] = 1
            size_deficit[i] = 2
            number_of_excess_not_covered[i] = 0

        else:

            size_excess[i] = 1
            size_deficit[i] = 1
            number_of_excess_not_covered[i] = 0

            mask[0, i] = False
            mask[1, i] = False

            max_height_array[i] = (starts_excess[i, 0] + energy_excess[i, 0])

    return (e_counter, d_counter, mask, max_height_array,
            capacity_excess, capacity_deficit,
            size_excess, size_deficit, number_of_excess_not_covered,
            starts_excess, starts_deficit,
            energy_excess, energy_deficit)

# TODO: AKTUELL KEIN RESIZE IMPLEMENTIERT BEI ZU KLEINEN ARRAYS  (passiert aber quasi eh nie)

@njit(nogil = True)
def process_phases(excess_array, deficit_array, start_times):

    # Provides the initial states for each Phase object and balances them
    (e_counter, d_counter, mask, max_height_array,
     capacity_excess, capacity_deficit,
     size_excess, size_deficit, number_of_excess_not_covered,
     starts_excess, starts_deficit,
     energy_excess, energy_deficit,
     ) = init(excess_array, deficit_array, start_times)

    # Return when we either start with no Excess or no Deficit
    if e_counter == 0 or d_counter == 0:
        return \
            (
                size_excess, size_deficit,
                starts_excess, starts_deficit,
                energy_excess, energy_deficit,
                mask
            )

    # start with an excess overflow right away
    idx = get_next_excess_index( 0, mask)

    while True:

        # Stop when either no more Excesses to move or no more Deficits to fill
        if e_counter == 0 or d_counter == 0:
            break

        # For each Phase there are 3 possibilities

        #1. Excess > Deficit
        next_phase_idx = get_next_non_balanced_phase(idx, mask)

        # Moves the Excess from the current Phase to the next non perfectly balanced phase
        e_counter, d_counter = move_excess(
            idx, next_phase_idx,
            max_height_array, mask,
            e_counter, d_counter,
            size_excess,
            number_of_excess_not_covered,
            starts_excess, energy_excess,
            size_deficit, starts_deficit, energy_deficit
        )

        # Stop when either no more Excesses to move or no more Deficits to fill
        if e_counter == 0 or d_counter == 0:
            break

        #2. Excess = Deficit (cant happen)
        # Nothing to move here

        #3. Excess < Deficit (cant happen)
        # Nothing to move here

        # Index goes to the next Excess
        idx = get_next_excess_index(idx, mask)

    return \
    (
        size_excess, size_deficit,
        starts_excess, starts_deficit,
        energy_excess, energy_deficit,
        mask
    )

