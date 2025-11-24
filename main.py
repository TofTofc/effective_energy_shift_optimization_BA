import importlib
import importlib.util
import os
import uuid
import time
import numpy as np

from helper.json_methodes import save_to_json, load_config, init_results_folders, get_run_info_from_json
from helper.plot_methodes import  plot_from_json
from helper.compare_methodes import test_result
from helper.runtime_fitting_methodes import log_log_linear_regression

delim = "-"*100

def import_module(folder_name: str):
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, "versions", folder_name, "effective_energy_shift.py")

    module_name = f"effective_energy_shift_{folder_name}_{uuid.uuid4().hex}"

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def init(worst_case_scenario: bool,
         master_seed: int,
         phase_count: int,
         repetition_count: int) \
         -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """
    Initialize energy excess/deficit arrays and starting time of the phases.

    Notes
    -----
    Worst-case scenario:
    - Excess from phase_count to 1
    - Deficit from 1 to phase_count

    Average-case scenario:
    - Excess and deficit values are randomly generated (0 - 100, integers) and independent
    - Seeds are unique for each (phase_count, repetition_index) combination

    Parameters
    ----------
    worst_case_scenario : bool
        True: deterministic worst case False: random average case.
    master_seed : int
        Base seed used.
    phase_count : int
        Length of each generated array (Number of Phases).
    repetition_count : int
        Number of repetitions: one excess/deficit pair per repetition.

    Returns
    -------
    energy_excess_list : list[np.ndarray]
        List of initial energy excess arrays, one array per repetition.
    energy_deficit_list : list[np.ndarray]
        List of initial energy deficit arrays, one array per repetition.
    start_time_phases : np.ndarray
        Array of phase start times (0,...,phase_count-1).
    """

    start_time_phases = np.arange(phase_count)

    energy_excess_list = []
    energy_deficit_list = []

    for rep_index in range(repetition_count):

        if worst_case_scenario:
            energy_excess = np.arange(phase_count, 0, -1)
            energy_deficit = np.arange(1, phase_count + 1)
        else:

            # Unique for combination of master seed, phase count and repetition index
            # Different for Excess (added 0) and Deficit (added 1)
            ss_excess = np.random.SeedSequence([master_seed, phase_count, rep_index, 0])
            ss_deficit = np.random.SeedSequence([master_seed, phase_count, rep_index, 1])

            rng_excess = np.random.default_rng(ss_excess)
            rng_deficit = np.random.default_rng(ss_deficit)

            energy_excess = rng_excess.integers(0, 100, phase_count)
            energy_deficit = rng_deficit.integers(0, 100, phase_count)

        energy_excess_list.append(energy_excess)
        energy_deficit_list.append(energy_deficit)

    return energy_excess_list, energy_deficit_list, start_time_phases


def output_runtime(module, total_runtime: float, repetition_count):
    full_name = module.__name__
    short_name = full_name[len("effective_energy_shift_"):]
    short_name = "_".join(short_name.split("_")[:-1])

    print(f"Module: {short_name}, Mean runtime: {total_runtime:.8f}s")


def do_normal_mode(module, energy_excess_list, energy_deficit_list, start_time_phases, repetition_count, fake_run):

    runtimes_single = []
    module_results = []

    for i in range(repetition_count):

        start = time.perf_counter()
        result_dict = module.process_phases(energy_excess_list[i], energy_deficit_list[i], start_time_phases)
        end = time.perf_counter()

        runtimes_single.append(end - start)
        module_results.append(result_dict)

    median_runtime = np.median(runtimes_single)

    if not fake_run:
        output_runtime(module, median_runtime, repetition_count)

    return module_results, median_runtime


def main():

    cfg = load_config()

    while True:

        version_name, pending_phase_counts, repetition_count, master_seed, worst_case_scenario = get_run_info_from_json(cfg)

        if not pending_phase_counts:
            print("Job done. Everything was measured")
            break

        module = import_module(version_name)

        # Fake run for numba compiling
        energy_excess_lists, energy_deficit_lists, start_time_phases = init(worst_case_scenario, master_seed, 1,repetition_count)
        do_normal_mode(module, energy_excess_lists, energy_deficit_lists, start_time_phases, repetition_count=1, fake_run=True)

        print(f"{delim}\n{delim}")
        print(f"Working on version: {version_name}, "f"phase counts from {pending_phase_counts[0]} to {pending_phase_counts[-1]}")
        print(delim)

        for phase_count in pending_phase_counts:

            print(delim)
            print("Current Phase Count: ", phase_count)

            energy_excess_lists, energy_deficit_lists, start_time_phases = init(worst_case_scenario, master_seed, phase_count, repetition_count)

            module_results, median_runtime = do_normal_mode(module, energy_excess_lists, energy_deficit_lists, start_time_phases, repetition_count, fake_run=False)

            # TODO: Redo TESTS OF RESULTS
            #test_result(module_results)

            save_to_json(cfg, phase_count, median_runtime, version_name)


if __name__ == '__main__':

    cfg = load_config()

    init_results_folders(cfg)

    main()

    #plot_from_json(cfg)

    #log_log_linear_regression(cfg)