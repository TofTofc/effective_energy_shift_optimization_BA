import numpy as np
from versions.append_improved_init_capacity_1 import efes_dataclasses

def process_callback(callback, current_step, phases, mask, **kwargs):
    if callback is not None:
        stop_algorithm = callback(current_step=current_step, phases=phases, mask=mask, **kwargs)
        if isinstance(stop_algorithm, bool):
            return stop_algorithm
    return False

def balance_phase(phase: efes_dataclasses.Phase):

    start_max = max(phase.starts_excess[phase.size_excess - 1], phase.starts_deficit[phase.size_deficit - 1])

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

    # phase.energy_excess[current_phase.size - 1] < phase.energy_deficit[current_phase.size - 1]
    # debug('not enough excess -> balance and add new deficit')

    new_start = phase.starts_excess[phase.size_excess - 1] + phase.energy_excess[phase.size_excess - 1]
    energy_remaining = phase.energy_deficit[phase.size_deficit - 1] - phase.energy_excess[phase.size_excess - 1]

    phase.energy_deficit[phase.size_deficit - 1] = phase.energy_excess[phase.size_excess - 1]
    phase.deficit_balanced[phase.size_deficit - 1] = True

    phase.append_deficit(new_start, energy_remaining, False)

    return False, True


def balance_phases(phases, mask):
    if mask is None:
        mask = np.ones((2, len(phases)), dtype=bool)

    potential_balance = mask[0] & mask[1]

    mask[:, potential_balance] = np.array(list(map(balance_phase, phases[potential_balance]))).transpose()
    return phases, mask


def calculate_virtual_excess(current_phase, next_phase):
    overflow_content = current_phase.energy_excess[current_phase.size_excess - 1]
    overflow_start = current_phase.starts_excess[current_phase.size_excess - 1]

    blocking_excess_content = next_phase.energy_excess[next_phase.size_excess - 1]

    blocking_excess_start = next_phase.starts_excess[next_phase.size_excess - 1]

    virtual_excess_start = max(overflow_start, blocking_excess_start + blocking_excess_content)
    virtual_excess_content = overflow_content
    virtual_excess_id = current_phase.excess_ids[current_phase.size_excess - 1]

    return virtual_excess_start, virtual_excess_content, virtual_excess_id


def move_overflow(phases, mask, callback_between_steps:callable = None, callback_kwargs={}):

    add_virtual_excess_mask = np.roll(mask[0], shift=1)
    next_indices = (np.arange(len(mask[0]))[mask[0]] + 1) % len(mask[0])

    current_phases = phases[mask[0]]
    next_phases = phases[next_indices]

    virtual_excess = list(map(lambda args: calculate_virtual_excess(*args), zip(current_phases, next_phases)))

    # place virtual excess in next phase

    list(map(lambda args: args[0].append_excess(args[1][0], args[1][1], False, args[1][2]),
             zip(phases[next_indices], virtual_excess)))


    if process_callback(callback_between_steps, 'shift', phases, mask, **callback_kwargs):
        return phases, mask, True

    # remove excess at index -2 where we had excess (mask[0]) and where virtual excess has been added (np.roll(mask[0], shift=1))
    list(map(lambda phase:  phase.remove_excess(-2), phases[mask[0] & add_virtual_excess_mask]))


    # remove excess at index -1 where we had excess (mask[0]) and no virtual excess has been added (not np.roll(mask[0], shift=1))
    list(map(lambda phase: phase.remove_excess(-1), phases[mask[0] & ~add_virtual_excess_mask]))

    # print(phases)

    mask[0] = add_virtual_excess_mask

    if process_callback(callback_between_steps, 'settle', phases, mask, **callback_kwargs):
        return phases, mask, True

    return phases, mask, False


def process_phases(energy_excess: np.ndarray, energy_deficit: np.ndarray, start_time_phases,
                   verbose:bool = False,
                   callback_between_steps: callable = None,
                   callback_kwargs: dict = {}
                   ):

    phases = np.array([efes_dataclasses.Phase(excess, deficit, id=start_time_phase) for (excess, deficit, start_time_phase) in zip(energy_excess, energy_deficit, start_time_phases)])
    n_phases = len(phases)
    mask = None

    if process_callback(callback_between_steps, 'init', phases, mask, **callback_kwargs):
        return dict(phases=phases, mask=mask)

    while True:
        phases, mask = balance_phases(phases, mask)
        if process_callback(callback_between_steps, 'balance', phases, mask, **callback_kwargs):
            return dict(phases=phases, mask=mask)

        if verbose:
            print(f'{n_phases - np.count_nonzero(mask, axis=1)} of {n_phases} done.')
        if np.any(~np.any(mask, axis=1)):
            break
        phases, mask, stop_algorithm = move_overflow(phases, mask, callback_between_steps=callback_between_steps, callback_kwargs=callback_kwargs)

    return phases