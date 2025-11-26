import numpy as np
from numba import njit
from numba.typed import List

from versions.new_design import efes_dataclasses


@njit
def balance_phase(phase: efes_dataclasses.Phase):

    start_max = max(phase.starts_excess[phase.size_excess - 1],phase.starts_deficit[phase.size_deficit - 1])

    phase.starts_excess[phase.size_excess - 1] = start_max
    phase.starts_deficit[phase.size_deficit - 1] = start_max

    phase.excess_balanced[phase.size_excess - 1] = True

    if phase.energy_excess[phase.size_excess - 1] == phase.energy_deficit[phase.size_deficit - 1]:
        phase.deficit_balanced[phase.size_deficit - 1] = True
        return False, False

    if phase.energy_excess[phase.size_excess - 1] > phase.energy_deficit[phase.size_deficit - 1]:
        phase.deficit_balanced[phase.size_deficit - 1] = True

        new_start = phase.starts_deficit[phase.size_deficit - 1] + phase.energy_deficit[phase.size_deficit - 1]
        energy_remaining = phase.energy_excess[phase.size_excess - 1] - phase.energy_deficit[phase.size_deficit - 1]

        phase.energy_excess[phase.size_excess - 1] = phase.energy_deficit[phase.size_deficit - 1]

        phase.append_excess(new_start, energy_remaining, False, phase.excess_ids[phase.size_excess - 1])

        return True, False

    new_start = phase.starts_excess[phase.size_excess - 1] + phase.energy_excess[phase.size_excess - 1]
    energy_remaining = phase.energy_deficit[phase.size_deficit - 1] - phase.energy_excess[phase.size_excess - 1]

    phase.energy_deficit[phase.size_deficit - 1] = phase.energy_excess[phase.size_excess - 1]
    phase.deficit_balanced[phase.size_deficit - 1] = True

    phase.append_deficit(new_start, energy_remaining, False)
    return False, True


@njit
def calculate_virtual_excess(current_phase, next_phase):

    overflow_content = current_phase.energy_excess[current_phase.size_excess - 1]
    overflow_start = current_phase.starts_excess[current_phase.size_excess - 1]

    blocking_excess_content = next_phase.energy_excess[next_phase.size_excess - 1]
    blocking_excess_start = next_phase.starts_excess[next_phase.size_excess - 1]

    virtual_excess_start = max(overflow_start, blocking_excess_start + blocking_excess_content)
    virtual_excess_content = overflow_content
    virtual_excess_id = current_phase.excess_ids[current_phase.size_excess - 1]

    return virtual_excess_start, virtual_excess_content, virtual_excess_id


@njit
def balance_phases_njit(phases, mask):

    n = len(phases)
    for i in range(n):
        if mask[0, i] and mask[1, i]:
            res0, res1 = balance_phase(phases[i])
            mask[0, i] = res0
            mask[1, i] = res1
    return phases, mask


@njit
def move_overflow_njit(phases, mask):
    n = len(phases)


    add_virtual_excess_mask = np.empty(n, dtype=np.bool_)
    for i in range(n):
        add_virtual_excess_mask[i] = mask[0, (i - 1) % n]


    count_excess = 0
    for i in range(n):
        if mask[0, i]:
            count_excess += 1

    next_indices = np.empty(count_excess, dtype=np.int64)
    j = 0
    for i in range(n):
        if mask[0, i]:
            next_indices[j] = (i + 1) % n
            j += 1

    current_phases = List()
    for i in range(n):
        if mask[0, i]:
            current_phases.append(phases[i])

    next_phases = List()
    for k in range(len(next_indices)):
        idx = next_indices[k]
        next_phases.append(phases[idx])

    m = len(current_phases)
    virtual_a = np.empty(m, dtype=np.float64)
    virtual_b = np.empty(m, dtype=np.float64)
    virtual_c = np.empty(m, dtype=np.int64)

    for i in range(m):
        a, b, c = calculate_virtual_excess(current_phases[i], next_phases[i])
        virtual_a[i] = a
        virtual_b[i] = b
        virtual_c[i] = c

    for i in range(m):
        next_phases[i].append_excess(virtual_a[i], virtual_b[i], False, virtual_c[i])

    for i in range(n):
        if mask[0, i] and add_virtual_excess_mask[i]:
            phases[i].remove_excess(-2)
        elif mask[0, i] and not add_virtual_excess_mask[i]:
            phases[i].remove_excess(-1)

    for i in range(n):
        mask[0, i] = add_virtual_excess_mask[i]

    return phases, mask, False


@njit
def process_phases_njit(phases_typed_list):

    n = len(phases_typed_list)
    mask = np.ones((2, n), dtype=np.bool_)

    while True:
        phases_typed_list, mask = balance_phases_njit(phases_typed_list, mask)

        row0_all_false = True
        row1_all_false = True
        for i in range(n):
            if mask[0, i]:
                row0_all_false = False
            if mask[1, i]:
                row1_all_false = False

        if row0_all_false or row1_all_false:
            break

        phases_typed_list, mask, _ = move_overflow_njit(phases_typed_list, mask)

    return phases_typed_list

def process_phases(energy_excess: np.ndarray, energy_deficit: np.ndarray, start_time_phases,
                   verbose: bool = False):
    phases_list = List()
    for ex, de, t in zip(energy_excess, energy_deficit, start_time_phases):
        phases_list.append(efes_dataclasses.Phase(ex, de, id=t))

    phases_out = process_phases_njit(phases_list)

    return phases_out
