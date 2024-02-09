import os.path

import numpy as np
import runlength as rl
from typing import Union, Optional, List
import math_energy_systems as mes
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


"""
CORE FUNCTIONS
"""

def runlength_encode(array_to_encode: np.ndarray, loop_around: bool = True):
    """
    A function that performs run length encoding and simplifies the encoded results by combining matching start and ends.
    :param array_to_encode: The array that shall be encoded
    :param loop_around: (optional) If set to False, no simplification will be performed. Defaults to True.
    :return: The run length encoding (starts, lengths and values) describing the original array.
    """
    n_time_steps = array_to_encode.size
    starts, lengths, values = rl.rlencode(array_to_encode)
    if len(values) == 1:
        starts = np.array([0, n_time_steps - 1])
        lengths = np.array([n_time_steps, 0])
        if values[0]:
            """All values are True"""
            return starts, lengths, np.array([values[0], 0])
        """All values are False"""
        return starts, lengths, np.array([False, True])

    if not loop_around:
        return starts, lengths, values

    """
    Set a correct start for looping
    If we have all 1, the solution is trivial -> we return full flooding
    If we have all 0, the solution is trivial -> we return a linear increase of flooding
    If the first part is a 0, put it at the back
    If the first and last part is a 1, put the back at front
    """
    if values[0] == 0 and values[-1] == 0:
        """If the first and last part is zero, append the front to the end"""
        # starts -= starts[1]  # keep the indices for later decoding
        lengths[-1] = lengths[-1] + lengths[0]
        starts = starts[1:]
        lengths = lengths[1:]
        values = values[1:]
    elif values[0] == 0:
        """If the first part is a 0, put it at the back"""
        starts = np.roll(starts, 1)
        lengths = np.roll(lengths, 1)
        values = np.roll(values, 1)
    elif values[0] == 1 and values[-1] == 1:
        """If the first and last part is one, append the end to the front"""
        # starts -= starts[1]  # keep the indices for later decoding
        lengths[0] = lengths[0] + lengths[-1]
        starts[0] = starts[-1]
        starts = starts[:-1]
        lengths = lengths[:-1]
        values = values[:-1]

    return starts, lengths, values


def calculate_initial_self_sufficiency_and_self_consumption(power_generation: Union[np.ndarray, float, int],
                                                            power_demand: Union[np.ndarray, float, int],
                                                            delta_time_step: float,
                                                            efficiency_direct_usage: float):
    time_total = None
    if isinstance(power_demand, float) or isinstance(power_demand, int):
        time_total = delta_time_step * power_generation.size
        energy_demand = mes.calculate_energy_from_constant_power(power=power_demand, time_total=time_total)
    else:
        time_total = delta_time_step * power_demand.size
        energy_demand = mes.calculate_energy_from_power_array(power_demand, delta_time_step)

    if isinstance(power_generation, float) or isinstance(power_generation, int):
        time_total = delta_time_step * power_demand.size
        energy_generation = mes.calculate_energy_from_constant_power(power=power_generation, time_total=time_total)
    else:
        time_total = delta_time_step * power_generation.size
        energy_generation = mes.calculate_energy_from_power_array(power_generation, delta_time_step)

    energy_used_generation = mes.calculate_used_generation_energy(power_generation=power_generation, power_demand=power_demand, delta_time_step=delta_time_step, efficiency_direct_usage=efficiency_direct_usage)
    energy_covered_demand = mes.calculate_covered_demand_energy(power_generation=power_generation, power_demand=power_demand, delta_time_step=delta_time_step, efficiency_direct_usage=efficiency_direct_usage)
    self_sufficiency = mes.calculate_self_sufficiency(energy_covered_demand=energy_covered_demand, energy_demand=energy_demand)
    self_consumption = mes.calculate_self_consumption(energy_used_generation=energy_used_generation, energy_generation=energy_generation)

    return dict(self_sufficiency_initial=self_sufficiency,
                self_consumption_initial=self_consumption,
                energy_used_generation=energy_used_generation,
                energy_covered_demand=energy_covered_demand,
                energy_demand=energy_demand,
                energy_generation=energy_generation,
                time_total=time_total
                )


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

def compute_battery_arrays_from_phases(phases: List[efes_dataclasses.Phase], efficiency_discharging: float):

    capacity_phases = []
    energy_additional_phases = []

    for phase in phases:
        capacity_phases.extend(phase.starts_deficit[phase.deficit_balanced])
        energy_additional_phases.extend(phase.energy_deficit[phase.deficit_balanced])

    capacity_phases = np.array(capacity_phases)
    energy_additional_phases = np.array(energy_additional_phases)

    capacity = np.unique(np.sort(np.array([capacity_phases, capacity_phases + energy_additional_phases]).flatten()))

    effectiveness = np.zeros(len(capacity))
    for phase in phases:
        for capacity_lower, capacity_upper in zip(phase.starts_deficit[phase.deficit_balanced], phase.starts_deficit[phase.deficit_balanced] + phase.energy_deficit[phase.deficit_balanced]):
            effectiveness[(capacity_lower <= capacity) & (capacity < capacity_upper)] += 1

    delta_capacity = np.diff(capacity)
    delta_energy_additional = effectiveness[:-1]*delta_capacity
    energy_additional = efficiency_discharging * np.array([0, *delta_energy_additional.cumsum()])
    return dict(capacity=capacity, energy_additional=energy_additional, effectiveness=effectiveness)


def calculate_energy_per_phase(power_residual_generation: np.ndarray, power_max_discharging: float, power_max_charging: float, efficiency_discharging: float, efficiency_charging: float, delta_time_step: float):
    results = dict()

    starts_zero, lengths_zero, values_zero = runlength_encode(power_residual_generation >= 0)
    results['starts_phases'] = starts_zero
    results['lengths_phases'] = lengths_zero
    results['values_phases'] = values_zero

    power_residual_generation_clipped = np.clip(power_residual_generation, -power_max_discharging, power_max_charging)
    results['power_residual_generation_clipped'] = power_residual_generation_clipped

    N_phases = int(starts_zero.shape[0] / 2)
    results['N_phases'] = N_phases

    energy_excess = np.zeros(shape=(N_phases,))
    energy_deficit = np.zeros(shape=(N_phases,))

    for n in range(0, starts_zero.shape[0], 2):
        generation = power_residual_generation_clipped[np.arange(starts_zero[n], starts_zero[n] + lengths_zero[n]) % power_residual_generation_clipped.size].sum() * delta_time_step
        demand = power_residual_generation_clipped[np.arange(starts_zero[n + 1], starts_zero[n + 1] + lengths_zero[n + 1]) % power_residual_generation_clipped.size].sum() * delta_time_step

        energy_excess[int(n / 2)] = generation
        energy_deficit[int(n / 2)] = -demand

    results['energy_excess_wo_efficiency'] = energy_excess.copy()
    results['energy_deficit_wo_efficiency'] = energy_deficit.copy()
    energy_excess = efficiency_charging * energy_excess
    energy_deficit = (1. / efficiency_discharging) * energy_deficit
    results['energy_excess'] = energy_excess
    results['energy_deficit'] = energy_deficit.copy()
    return results


def get_energy_over_x_arrays(values):
    sorted_values = np.sort(np.abs(values.astype(float)))
    sorted_values = sorted_values[sorted_values > 0]

    delta_sorted_values = np.diff(sorted_values, prepend=[0])
    slope = np.arange(delta_sorted_values.shape[0], 0, -1)

    energy = (slope*delta_sorted_values).cumsum()   # np.tril(slope).dot(delta_sorted_values), but faster
    mask = delta_sorted_values > 0

    x = np.array([0, *sorted_values[mask]])
    energy = np.array([0, *energy[mask]])
    energy_slope = slope[mask]
    return x, energy, energy_slope

def get_phase_power_data(power_residual_generation: np.ndarray, delta_time_step: float,
                         starts_phases: np.ndarray, lengths_phases: np.ndarray, values_phases: np.ndarray
                         ):
    """
    This function will analyse each individual excess and deficit phase and produces arrays, that can be later used to
    dimension the minimum required charging and discharging power of the storage system.

    The relevant data for dimensioning the power levels of each phase are PhaseData::energy_amounts and PhaseData::power.
    The power dimensioning is done in compute_min_powers_for_capacity

    :param power_residual_generation: The residual generation curve with positive excess and negative deficit values.
    :param delta_time_step: The fixed time step.
    :param starts_phases: The start indices of each excess and deficit phase.
    :param lengths_phases: The length of each excess and deficit phase.
    :param values_phases: The value of each excess (True) and deficit (False) phase.

    :return: A dict with the following entries:
        phase_data_deficit (List[PhaseData]): The data and information of the deficit phases.
        phase_data_excess (List[PhaseData]): The data and information of the excess phases.
    """
    starts_deficit = starts_phases[~values_phases]
    starts_excess = starts_phases[values_phases]
    lengths_deficit = lengths_phases[~values_phases]
    lengths_excess = lengths_phases[values_phases]

    def func(start, length):
        phase_data = efes_dataclasses.PhaseData()
        power_values = power_residual_generation[np.arange(start, start + length) % power_residual_generation.size]
        power, energy, energy_slope = get_energy_over_x_arrays(power_values)
        phase_data.power = power
        phase_data.energy = delta_time_step * energy
        phase_data.duration = delta_time_step * energy_slope
        return phase_data


    phase_data_deficit = list(map(lambda arg: func(*arg), zip(starts_deficit, lengths_deficit)))
    phase_data_excess = list(map(lambda arg: func(*arg), zip(starts_excess, lengths_excess)))
    return dict(phase_data_deficit=phase_data_deficit, phase_data_excess=phase_data_excess)


"""MAIN FUNCTIONS"""

def analyse_power_data(power_generation, power_demand, delta_time_step,
                       power_max_discharging=np.inf, power_max_charging=np.inf,
                       efficiency_direct_usage=1.0, efficiency_discharging=1.0, efficiency_charging=1.0,
                       callback_between_steps: callable = None,
                       callback_kwargs: dict = {}
                       ):
    """
    This is the frame of the Effective Energy Shift (EfES) algorithm, that will call the functions to perform
    the different steps for preparing the data, processing it and calculating the output of the analyis.

    :param power_generation: The generation power data for the analysis.
    :param power_demand: The demand power data for the analysis.
    :param delta_time_step: The timestep for the provided power data.
    :param power_max_discharging: (optional) The maximum discharging power of the EES. Defaults to np.inf
    :param power_max_charging: (optional) The maximum charging power of the EES. Defaults to np.inf
    :param efficiency_direct_usage: (optional) The efficiency for directly using the generation for the
            demand (without EES).  Defaults to 1.
    :param efficiency_discharging: (optional) The efficiency for discharging the EES. Defaults to 1.
    :param efficiency_charging: (optional) The efficiency for charging the EES. Defaults to 1.
    :return: An instance of efes_dataclasses.AnalysisResults containing the results like:
        - the total time span of the provided data
        - energy amounts for directly used generation and covered demand as well as the total energy generation and demand
        - the initial self-sufficiency and self-consumption
        - the residual generation data, clipped at the maximum charging and discharging power
        - the run length encoded information for starts, lengths and types (values) of excess and deficit phases
        - the total number of phases
        - the excess energy amounts with and without charging efficiency
        - the deficit energy amounts with and without discharging efficiency
        - the sorted set of all phases containing the excess and deficit packets after performing the algorithm
        - the effectiveness array
        - the array for capacity and additional energy
        - the self-sufficiency and self-consumption at those capacity values
        - the gain in total at those capacity values
        - the maximum useful capacity, the maximum reachable additional energy, self-sufficiency and self-consumption
    """
    power_covered_demand = mes.get_covered_demand_power(power_generation=power_generation, power_demand=power_demand, efficiency_direct_usage=efficiency_direct_usage)
    power_used_generation = mes.get_used_generation_power(power_generation=power_generation, power_demand=power_demand, efficiency_direct_usage=efficiency_direct_usage)
    power_residual_generation = power_generation - power_demand - (power_used_generation-power_covered_demand)

    try:
        _ = power_residual_generation[0]
    except TypeError as e:
        raise AttributeError(f'Either power_generation or power_demand have to be iterables (e.g. np.ndarray)! You provided {type(power_generation)=} and {type(power_demand)=}.')

    if all(power_residual_generation > 0):
        raise AttributeError(
            'The power data does not contain any negative values and there is no deficit therefore. An energy storage system will not provide any benefit.')

    if all(power_residual_generation < 0):
        raise AttributeError(
            'The power data does not contain any positive values and there is no excess therefore. An energy storage system will not be able to charge and will therefore not provide any benefit.')

    if delta_time_step == 0:
        raise AttributeError('delta_time_step is 0, which means no time progress and therefore no energy.')

    if power_max_discharging <= 0:
        raise AttributeError(
            'power_max_discharging is less or equal to 0, which means the energy system can not be discharged.')

    if power_max_charging <= 0:
        raise AttributeError(
            'power_max_charging is less or equal to 0, which means the energy system can not be charged.')

    if efficiency_direct_usage <= 0:
        raise AttributeError(
            'efficiency_direct_usage is less or equal to 0, which means there is no energy left in the system.')

    if efficiency_discharging <= 0:
        raise AttributeError(
            'efficiency_discharging is less or equal to 0, which means the energy system can not be discharged.')

    if efficiency_charging <= 0:
        raise AttributeError(
            'efficiency_charging is less or equal to 0, which means the energy system can not be charged.')

    data_input = efes_dataclasses.DataInput(
        power_generation=power_generation,
        power_demand=power_demand,
        power_used_generation=power_used_generation,
        power_covered_demand=power_covered_demand,
        power_residual_generation=power_residual_generation,
        delta_time_step=delta_time_step,
        power_max_discharging=power_max_discharging,
        power_max_charging=power_max_charging,
        efficiency_direct_usage=efficiency_direct_usage,
        efficiency_discharging=efficiency_discharging,
        efficiency_charging=efficiency_charging
    )

    analysis_results = efes_dataclasses.AnalysisResults(data_input=data_input)

    analysis_results.update(**calculate_initial_self_sufficiency_and_self_consumption(
        power_generation=data_input.power_generation,
        power_demand=data_input.power_demand,
        delta_time_step=data_input.delta_time_step,
        efficiency_direct_usage=data_input.efficiency_direct_usage
    ))


    analysis_results.update(**calculate_energy_per_phase(
        power_residual_generation=data_input.power_residual_generation,
        power_max_discharging=data_input.power_max_discharging,
        power_max_charging=data_input.power_max_charging,
        efficiency_discharging=data_input.efficiency_discharging,
        efficiency_charging=data_input.efficiency_charging,
        delta_time_step=data_input.delta_time_step
    ))

    analysis_results.update(**process_phases(
        energy_excess=analysis_results.energy_excess,
        energy_deficit=analysis_results.energy_deficit,
        start_time_phases=data_input.delta_time_step*analysis_results.starts_phases,
        callback_between_steps=callback_between_steps,
        callback_kwargs=callback_kwargs
    ))

    analysis_results.update(**compute_battery_arrays_from_phases(
        phases=analysis_results.phases,
        efficiency_discharging=data_input.efficiency_discharging
    ))

    analysis_results.capacity_max = analysis_results.capacity[-1]
    analysis_results.energy_additional_max = analysis_results.energy_additional[-1]

    analysis_results.gain = mes.calculate_gain_from_energy_and_capacity(energy_additional=analysis_results.energy_additional, capacity=analysis_results.capacity)

    analysis_results.self_sufficiency = mes.calculate_self_sufficiency_from_additional_energy(
        energy_additional=analysis_results.energy_additional,
        energy_demand=analysis_results.energy_demand,
        self_sufficiency_initial=analysis_results.self_sufficiency_initial
    )

    analysis_results.self_sufficiency_max = analysis_results.self_sufficiency_initial + analysis_results.energy_additional_max/analysis_results.energy_demand

    analysis_results.self_consumption = mes.calculate_self_consumption_from_additional_energy(
        energy_additional=analysis_results.energy_additional,
        energy_generation=analysis_results.energy_generation,
        self_consumption_initial=analysis_results.self_consumption_initial,
        efficiency_discharging=analysis_results.data_input.efficiency_discharging,
        efficiency_charging=analysis_results.data_input.efficiency_charging
    )

    analysis_results.self_consumption_max = analysis_results.self_consumption_initial + analysis_results.energy_additional_max/(data_input.efficiency_charging*data_input.efficiency_discharging*analysis_results.energy_generation)
    return analysis_results


def run_query(analysis_results: efes_dataclasses.AnalysisResults, query_results: efes_dataclasses.QueryResults):
    """
    Complete the query results by filling in missing values.
    :param analysis_results: The analysis results from the EfES algorithm.
    :param query_results: The query results either from one of the run_dimensioning_query_for_target_{...} functions.
    :return: The query results filled up by the missing values.
    """
    self_sufficiency_target = query_results.query_input.self_sufficiency_target
    self_consumption_target = query_results.query_input.self_consumption_target
    energy_additional_target = query_results.query_input.energy_additional_target
    capacity_target = query_results.query_input.capacity_target

    if self_sufficiency_target is not None:
        # add self_sufficiency
        query_results.self_sufficiency = np.clip(a=self_sufficiency_target, a_min=analysis_results.self_sufficiency_initial, a_max=analysis_results.self_sufficiency_max)

        # calculate self_consumption
        query_results.self_consumption = mes.calculate_self_consumption_from_self_sufficiency(
            self_sufficiency=query_results.self_sufficiency,
            energy_demand=analysis_results.energy_demand,
            energy_generation=analysis_results.energy_generation,
            self_consumption_initial=analysis_results.self_consumption_initial,
            self_sufficiency_initial=analysis_results.self_sufficiency_initial,
            efficiency_discharging=analysis_results.data_input.efficiency_discharging,
            efficiency_charging=analysis_results.data_input.efficiency_charging
        )

        # calculate energy_additional
        query_results.energy_additional = mes.calculate_additional_energy_from_self_sufficiency(
            self_sufficiency=query_results.self_sufficiency,
            self_sufficiency_initial=analysis_results.self_sufficiency_initial,
            energy_demand=analysis_results.energy_demand
        )

        # calculate capacity
        query_results.capacity = mes.calculate_capacity_from_additional_energy(
            energy_additional=query_results.energy_additional,
            energy_additional_array=analysis_results.energy_additional,
            capacity_array=analysis_results.capacity
        )

    elif self_consumption_target is not None:
        # add self_consumption
        query_results.self_consumption = np.clip(a=self_consumption_target, a_min=analysis_results.self_consumption_initial, a_max=analysis_results.self_consumption_max)

        # calculate self_sufficiency
        query_results.self_sufficiency = mes.calculate_self_sufficiency_from_self_consumption(
            self_consumption=query_results.self_consumption,
            energy_demand=analysis_results.energy_demand,
            energy_generation=analysis_results.energy_generation,
            self_consumption_initial=analysis_results.self_consumption_initial,
            self_sufficiency_initial=analysis_results.self_sufficiency_initial,
            efficiency_discharging=analysis_results.data_input.efficiency_discharging,
            efficiency_charging=analysis_results.data_input.efficiency_charging
        )

        # calculate energy_additional
        query_results.energy_additional = mes.calculate_additional_energy_from_self_sufficiency(
            self_sufficiency=query_results.self_sufficiency,
            energy_demand=analysis_results.energy_demand,
            self_sufficiency_initial=analysis_results.self_sufficiency_initial
        )

        # calculate capacity
        query_results.capacity = mes.calculate_capacity_from_additional_energy(
            energy_additional=query_results.energy_additional,
            energy_additional_array=analysis_results.energy_additional,
            capacity_array=analysis_results.capacity
        )

    elif energy_additional_target is not None:
        # add energy_additional
        query_results.energy_additional = np.clip(a=energy_additional_target, a_min=0., a_max=analysis_results.energy_additional_max)

        # calculate self_sufficiency
        query_results.self_sufficiency = mes.calculate_self_sufficiency_from_additional_energy(
            energy_additional=query_results.energy_additional,
            energy_demand=analysis_results.energy_demand,
            self_sufficiency_initial=analysis_results.self_sufficiency_initial
        )

        # calculate self_consumption
        query_results.self_consumption = mes.calculate_self_consumption_from_additional_energy(
            energy_additional=query_results.energy_additional,
            energy_generation=analysis_results.energy_generation,
            self_consumption_initial=analysis_results.self_consumption_initial,
            efficiency_charging=analysis_results.data_input.efficiency_charging,
            efficiency_discharging=analysis_results.data_input.efficiency_discharging
        )

        # calculate capacity
        query_results.capacity = mes.calculate_capacity_from_additional_energy(
            energy_additional=query_results.energy_additional,
            energy_additional_array=analysis_results.energy_additional,
            capacity_array=analysis_results.capacity
        )

    elif capacity_target is not None:
        # add capacity
        query_results.capacity = np.clip(a=capacity_target, a_min=0., a_max=None)

        # calculate energy_additional
        query_results.energy_additional = mes.calculate_additional_energy_from_capacity(
            capacity=query_results.capacity,
            energy_additional_array=analysis_results.energy_additional,
            capacity_array=analysis_results.capacity
        )

        # calculate self_sufficiency
        query_results.self_sufficiency = mes.calculate_self_sufficiency_from_additional_energy(
            energy_additional=query_results.energy_additional,
            energy_demand=analysis_results.energy_demand,
            self_sufficiency_initial=analysis_results.self_sufficiency_initial
        )

        # calculate self_consumption
        query_results.self_consumption = mes.calculate_self_consumption_from_additional_energy(
            energy_additional=query_results.energy_additional,
            energy_generation=analysis_results.energy_generation,
            self_consumption_initial=analysis_results.self_consumption_initial,
            efficiency_discharging=analysis_results.data_input.efficiency_discharging,
            efficiency_charging=analysis_results.data_input.efficiency_charging
        )


    # compute gain
    query_results.gain = mes.calculate_gain_from_energy_and_capacity(
        energy_additional=query_results.energy_additional,
        capacity=query_results.capacity
    )

    # compute gain per 24h
    query_results.gain_per_day = mes.calculate_gain_per_day(
        gain=query_results.gain,
        time_total=analysis_results.time_total
    )
    return query_results

def run_dimensioning_query_for_target_self_sufficiency(analysis_results: efes_dataclasses.AnalysisResults, self_sufficiency_target: Union[np.array, float]):
    """
    Query the analysis results at specific values for the self-sufficiency.
    :param analysis_results: The analysis results from the EfES algorithm.
    :param self_sufficiency_target: Specific values for the self-sufficiency, that will be sampled from the analysis results.
    :return: An instance of efes_dataclasses.QueryResults
    """
    query_input = efes_dataclasses.QueryInput(self_sufficiency_target=self_sufficiency_target)
    query_results = efes_dataclasses.QueryResults(query_input=query_input)
    query_results = run_query(analysis_results=analysis_results, query_results=query_results)
    return query_results

def run_dimensioning_query_for_target_self_consumption(analysis_results: efes_dataclasses.AnalysisResults, self_consumption_target: Union[np.array, float]):
    """
    Query the analysis results at specific values for the self-consumption.
    :param analysis_results: The analysis results from the EfES algorithm.
    :param self_consumption_target: Specific values for the self-consumption, that will be sampled from the analysis results.
    :return: An instance of efes_dataclasses.QueryResults
    """
    query_input = efes_dataclasses.QueryInput(self_consumption_target=self_consumption_target)
    query_results = efes_dataclasses.QueryResults(query_input=query_input)
    query_results = run_query(analysis_results=analysis_results, query_results=query_results)
    return query_results

def run_dimensioning_query_for_target_additional_energy(analysis_results: efes_dataclasses.AnalysisResults, energy_additional_target: Union[np.array, float]):
    """
    Query the analysis results at specific values for the additional energy.
    :param analysis_results: The analysis results from the EfES algorithm.
    :param energy_additional_target: Specific values for the additional energy, that will be sampled from the analysis results.
    :return: An instance of efes_dataclasses.QueryResults
    """
    query_input = efes_dataclasses.QueryInput(energy_additional_target=energy_additional_target)
    query_results = efes_dataclasses.QueryResults(query_input=query_input)
    query_results = run_query(analysis_results=analysis_results, query_results=query_results)
    return query_results

def run_dimensioning_query_for_target_capacity(analysis_results: efes_dataclasses.AnalysisResults, capacity_target: Union[np.array, float]):
    """
    Query the analysis results at specific values for the capacity.
    :param analysis_results: The analysis results from the EfES algorithm.
    :param capacity_target: Specific values for the capacity, that will be sampled from the analysis results.
    :return: An instance of efes_dataclasses.QueryResults
    """
    query_input = efes_dataclasses.QueryInput(capacity_target=capacity_target)
    query_results = efes_dataclasses.QueryResults(query_input=query_input)
    query_results = run_query(analysis_results=analysis_results, query_results=query_results)
    return query_results


def perform_energy_storage_dimensioning(power_generation, power_demand, delta_time_step,
                                        power_max_discharging=np.inf, power_max_charging=np.inf,
                                        efficiency_direct_usage=1.0, efficiency_discharging=1.0, efficiency_charging=1.0,
                                        self_sufficiency_target=None, self_consumption_target=None, energy_additional_target=None, capacity_target=None,
                                        callback_between_steps: callable = None,
                                        callback_kwargs: dict = {}
                                        ):
    """
    The main function that will run the EfES algorithm based on the provided data.
    If any of the {...}_target parameters has been provided it will query the results at those data points.

    :param power_generation: The generation power data for the analysis.
    :param power_demand: The demand power data for the analysis.
    :param delta_time_step: The timestep for the provided power data.
    :param power_max_discharging: (optional) The maximum discharging power of the EES. Defaults to np.inf
    :param power_max_charging: (optional) The maximum charging power of the EES. Defaults to np.inf
    :param efficiency_direct_usage: (optional) The efficiency for directly using the generation for the
            demand (without EES).  Defaults to 1.
    :param efficiency_discharging: (optional) The efficiency for discharging the EES. Defaults to 1.
    :param efficiency_charging: (optional) The efficiency for charging the EES. Defaults to 1.
    :param self_sufficiency_target: (optional) If provided, the analysis results will be sampled at the specified
            values. Those query results are added to the analysis results.
    :param self_consumption_target: (optional) If provided, the analysis results will be sampled at the specified
            values. Those query results are added to the analysis results.
    :param energy_additional_target: (optional) If provided, the analysis results will be sampled at the specified
            values. Those query results are added to the analysis results.
    :param capacity_target: (optional) If provided, the analysis results will be sampled at the specified values. Those
            query results are added to the analysis results.
    :return: All results with an instance of efes_dataclasses.Results.
    """
    results = efes_dataclasses.Results()

    try:
        results.analysis_results = analyse_power_data(
            power_generation=power_generation,
            power_demand=power_demand,
            delta_time_step=delta_time_step,
            power_max_discharging=power_max_discharging,
            power_max_charging=power_max_charging,
            efficiency_direct_usage=efficiency_direct_usage,
            efficiency_discharging=efficiency_discharging,
            efficiency_charging=efficiency_charging,
            callback_between_steps=callback_between_steps,
            callback_kwargs=callback_kwargs
        )
    except AttributeError as e:
        print('An error has been raised by analyse_power_data and the dimensioning can not be performed!')
        raise (e)

    if self_sufficiency_target is not None:
        if results.query_results is None:
            results.query_results = []

        try:
            results.query_results.append(run_dimensioning_query_for_target_self_sufficiency(
                analysis_results=results.analysis_results,
                self_sufficiency_target=self_sufficiency_target
            ))
        except AttributeError as e:
            print('An error has been raised by run_dimensioning_query_for_target_self_sufficiency. You still get the analysis results.')
            print(e)
            results.query_results.append(e)

    if self_consumption_target is not None:
        if results.query_results is None:
            results.query_results = []

        try:
            results.query_results.append(run_dimensioning_query_for_target_self_consumption(
                analysis_results=results.analysis_results,
                self_consumption_target=self_consumption_target
            ))
        except AttributeError as e:
            print('An error has been raised by run_dimensioning_query_for_target_self_consumption. You still get the analysis results.')
            print(e)
            results.query_results.append(e)

    if energy_additional_target is not None:
        if results.query_results is None:
            results.query_results = []
        try:
            results.query_results.append(run_dimensioning_query_for_target_additional_energy(
                analysis_results=results.analysis_results,
                energy_additional_target=energy_additional_target
            ))
        except AttributeError as e:
            print('An error has been raised by run_dimensioning_query_for_target_additional_energy. You still get the analysis results.')
            print(e)
            results.query_results.append(e)

    if capacity_target is not None:
        if results.query_results is None:
            results.query_results = []
        try:
            results.query_results.append(run_dimensioning_query_for_target_capacity(
                analysis_results=results.analysis_results,
                capacity_target=capacity_target
            ))
        except AttributeError as e:
            print('An error has been raised by run_dimensioning_query_for_target_capacity. You still get the analysis results.')
            print(e)
            results.query_results.append(e)

    return results

def run_parameter_study(power_generation, power_demand, delta_time_step, parameter_variation, result_dir=None, **kwargs):
    """
    Run a parameter study on a number of parameters. This can also be obligatory parameters, like power_generation.
    :param power_generation: The generation power data used for the analysis.
    :param power_demand: The demand power data used for the analysis.
    :param delta_time_step: The timestep of the provided data arrays.
    :param parameter_variation: A dictionary with the name of a parameter as keys and list of equal length for the
            variation. All list will be iterated in parallel to get the parameters for a single analysis.
    :param result_dir: (optional) A path to a directory that is used to store the intermediate and final results. Defaults to 'results_parameter_study'.
    :param kwargs: (optional) Other key-word arguments that will be passed to perform_energy_storage_dimensioning(...)
    :return: An instance of efes_dataclasses.ParameterStudyResults containing among other paths to intermediate results.
    """
    '''Create a subfolder to store the results, since it will allow us to free RAM.'''
    if result_dir is None:
        result_dir = 'results_parameter_study'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    parameter_study_results = efes_dataclasses.ParameterStudyResults(parameter_variation=parameter_variation)

    '''Create the basecases.csv file'''
    parameter_variation.to_csv(f'{result_dir}/basecases.csv', index_label='basecase')

    parameter_study_results.results = [None]*parameter_variation.shape[0]

    def run_single_varition(variation):
        kwargs.update(variation.to_dict())
        basecase = kwargs['basecase']
        del kwargs['basecase']
        if not os.path.exists(f'{result_dir}/{basecase}'):
            os.makedirs(f'{result_dir}/{basecase}')

        filename_results = f'{result_dir}/{basecase}/results.pickle'
        if os.path.exists(filename_results):
            print(f'Loading from {filename_results}')
            return filename_results

        print(f'Running: {variation.to_dict()}')
        results = perform_energy_storage_dimensioning(power_generation, power_demand, delta_time_step, **kwargs)
        efes_dataclasses.pickle(results, filename_results)
        del results
        return filename_results

    parameter_variation['basecase'] = parameter_variation.index
    basecase_max = parameter_variation['basecase'].max()
    parameter_variation['basecase'] = parameter_variation['basecase'].map(lambda basecase: f'{basecase:0{len(str(basecase_max))}d}')

    parameter_study_results.results = parameter_variation.apply(run_single_varition, axis=1).to_list()
    print('All finished')
    return parameter_study_results
