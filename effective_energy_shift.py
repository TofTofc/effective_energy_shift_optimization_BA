import os.path

import numpy as np
from typing import Union, Optional, List
import efes_dataclasses

#import logging
#logging.basicConfig(filename='ebd.log', encoding='utf-8', level=logging.ERROR)

"""
GENERAL FUNCTIONS
"""

def descr() -> str:
    s = """
    This script will run the Effective Energy Shift (EfES) algorithm on a provided generation power array and one for the demand power.
    """
    return s


def get_scaling(num: Union[np.ndarray, float, int]):
    try:
        return list(map(lambda v: get_scaling(v), num))
    except:
        pass

    if num == 0:
        return 0, ''
    if abs(num) > 1e12:
        return 1e-12, 'T'
    if abs(num) > 1e9:
        return 1e-9, 'G'
    if abs(num) > 1e6:
        return 1e-6, 'M'
    if abs(num) > 1e3:
        return 1e-3, 'k'
    if abs(num) < 1e-6:
        return 1e6, 'u'
    if abs(num) < 1e-3:
        return 1e3, 'm'
    return 1., ''

def get_num_from_str_with_scale_and_unit(num_str: Union[np.array, str], unit:str):
    if isinstance(num_str, list):
        return list(map(lambda v: get_num_from_str_with_scale_and_unit(v, unit), num_str))

    num, scale_and_unit = num_str.split(' ')

    decimal_scaling = 10**len(num.split('.')[1])
    num = float(num)
    scale_str = scale_and_unit.replace(unit, '')
    scales = dict(k=1e3, M=1e6, G=1e9, T=1e12, m=1e-3, u=1e-6, n=1e-9)
    scale = 1. if scale_str == '' else scales[scale_str]
    num_scaled_back = scale * np.round(num * decimal_scaling) / decimal_scaling
    return num_scaled_back

def pretty_print(num: Union[np.ndarray, float, int], unit: str, decimals: int = 2):
    """
    A function that takes a numeric value or a list of numeric values, a unit string and an optional decimal count.
    It will use the most suitable scaling to express the number with the unit as a string.
    Examples:
        - pretty_print(num=10000, unit='W', decimals=2)  -->  '10.00 kW'
        - pretty_print(num=0.023568, unit='N', decimals=1)  -->  '23.6 mN'
        - pretty_print(num=[5, 600, 2300, 3650000], unit='Wh')  -->  ['5.00 Wh', '600.00 Wh', '2.30 kWh', '3.65 MWh']

    :param num: The numeric value or the array of numeric values
    :param unit: The unit string that should be used as the suffix
    :param decimals: (optional) The decimal count that should be used for formatting.
    :return: A formatted string or a list of formatted strings.
    """
    try:
        return list(map(lambda v: pretty_print(v, unit, decimals), num))
    except:
        pass

    scaling_factor, scaling_str = get_scaling(num)
    return f'{num * scaling_factor:.{decimals}f} {scaling_str}{unit}'

def process_callback(callback, current_step, phases, mask, **kwargs):
    if callback is not None:
        stop_algorithm = callback(current_step=current_step, phases=phases, mask=mask, **kwargs)
        if isinstance(stop_algorithm, bool):
            return stop_algorithm
    return False


def balance_phase(phase: efes_dataclasses.Phase):
    #logging.info(f'Balancing phase {phase.id}')

    start_max = max(phase.starts_excess[-1], phase.starts_deficit[-1])
    phase.starts_excess[-1] = start_max
    phase.starts_deficit[-1] = start_max

    phase.excess_balanced[-1] = True

    if phase.energy_excess[-1] == phase.energy_deficit[-1]:
        #logging.info('excess matches deficit -> balance')
        phase.deficit_balanced[-1] = True
        return False, False

    if phase.energy_excess[-1] > phase.energy_deficit[-1]:
        #logging.info('more excess than needed -> balance and add new excess')
        phase.deficit_balanced[-1] = True

        new_start = phase.starts_deficit[-1] + phase.energy_deficit[-1]
        energy_remaining = phase.energy_excess[-1] - phase.energy_deficit[-1]

        phase.energy_excess[-1] = phase.energy_deficit[-1]

        phase.starts_excess = np.append(phase.starts_excess, new_start)
        phase.energy_excess = np.append(phase.energy_excess, energy_remaining)
        phase.excess_balanced = np.append(phase.excess_balanced, False)
        phase.excess_ids = np.append(phase.excess_ids, phase.excess_ids[-1])
        return True, False

    # phase.energy_excess[-1] < phase.energy_deficit[-1]
    # debug('not enough excess -> balance and add new deficit')

    new_start = phase.starts_excess[-1] + phase.energy_excess[-1]
    energy_remaining = phase.energy_deficit[-1] - phase.energy_excess[-1]

    phase.energy_deficit[-1] = phase.energy_excess[-1]
    phase.deficit_balanced[-1] = True

    phase.starts_deficit = np.append(phase.starts_deficit, new_start)
    phase.energy_deficit = np.append(phase.energy_deficit, energy_remaining)
    phase.deficit_balanced = np.append(phase.deficit_balanced, False)
    return False, True


def balance_phases(phases, mask):
    if mask is None:
        mask = np.ones((2, len(phases)), dtype=bool)

    potential_balance = mask[0] & mask[1]

    mask[:, potential_balance] = np.array(list(map(balance_phase, phases[potential_balance]))).transpose()
    return phases, mask


def calculate_virtual_excess(current_phase, next_phase):
    overflow_content = current_phase.energy_excess[-1]
    overflow_start = current_phase.starts_excess[-1]

    blocking_excess_content = next_phase.energy_excess[-1]
    blocking_excess_start = next_phase.starts_excess[-1]

    virtual_excess_start = max(overflow_start, blocking_excess_start + blocking_excess_content)
    virtual_excess_content = overflow_content
    virtual_excess_id = current_phase.excess_ids[-1]

    return virtual_excess_start, virtual_excess_content, virtual_excess_id

def add_excess_to_phase(phase, excess_start, excess_content, excess_id):
    phase.starts_excess = np.append(phase.starts_excess, excess_start)
    phase.energy_excess = np.append(phase.energy_excess, excess_content)
    phase.excess_balanced = np.append(phase.excess_balanced, False)
    phase.excess_ids = np.append(phase.excess_ids, excess_id)


def remove_excess(phase, index_to_remove):
    phase.energy_excess = np.delete(phase.energy_excess, obj=index_to_remove)
    phase.starts_excess = np.delete(phase.starts_excess, obj=index_to_remove)
    phase.excess_balanced = np.delete(phase.excess_balanced, obj=index_to_remove)
    phase.excess_ids = np.delete(phase.excess_ids, obj=index_to_remove)

def move_overflow(phases, mask, callback_between_steps:callable = None, callback_kwargs={}):
    #logging.info(f'overflow in {np.nonzero(mask[0])}')
    add_virtual_excess_mask = np.roll(mask[0], shift=1)
    next_indices = (np.arange(len(mask[0]))[mask[0]] + 1) % len(mask[0])

    current_phases = phases[mask[0]]
    next_phases = phases[next_indices]

    virtual_excess = list(map(lambda args: calculate_virtual_excess(*args), zip(current_phases, next_phases)))

    # place virtual excess in next phase
    list(map(lambda args: add_excess_to_phase(args[0], *args[1]), zip(phases[next_indices], virtual_excess)))

    #logging.info(f'Excess added to {next_indices}')

    if process_callback(callback_between_steps, 'shift', phases, mask, **callback_kwargs):
        return phases, mask, True

    # remove excess at index -2 where we had excess (mask[0]) and where virtual excess has been added (np.roll(mask[0], shift=1))
    list(map(lambda phase: remove_excess(phase, -2), phases[mask[0] & add_virtual_excess_mask]))
    #logging.info(f'Excess at -2 removed from {np.nonzero(mask[0] & add_virtual_excess_mask)}')

    # remove excess at index -1 where we had excess (mask[0]) and no virtual excess has been added (not np.roll(mask[0], shift=1))
    list(map(lambda phase: remove_excess(phase, -1), phases[mask[0] & ~add_virtual_excess_mask]))
    #logging.info(f'Excess at -1 removed from {np.nonzero(mask[0] & ~add_virtual_excess_mask)}')
    # print(phases)

    mask[0] = add_virtual_excess_mask
    #logging.info(f'new excess in {mask[0]}')
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

    return dict(phases=phases, mask=mask)