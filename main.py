import cProfile
import importlib
import importlib.util
import os
import pstats
import sys
import uuid
import time
import numpy as np
from matplotlib import pyplot as plt

from helper.json_methodes import save_to_json, load_config, create_result_folders_and_init_json
from helper.plot_methodes import plot_current_run, plot_from_json
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


def get_modules(indices: list[int], versions: list):
    for i in indices:
        if i < 0 or i >= len(versions):
            sys.exit(f"Indices not suitable")

    modules = [import_module(versions[i]) for i in indices]
    return modules


def output_runtime(module, total_runtime: float, repetition_count):
    full_name = module.__name__
    short_name = full_name[len("effective_energy_shift_"):]
    short_name = "_".join(short_name.split("_")[:-1])

    print(f"Module: {short_name}, Mean runtime: {total_runtime:.8f}s")


def do_submethod_analysis(module, energy_excess_list, energy_deficit_list, start_time_phases):
    profile = cProfile.Profile()
    profile.runcall(module.process_phases, energy_excess_list[0], energy_deficit_list[0], start_time_phases)

    ps = pstats.Stats(profile).sort_stats("cumtime")
    ps.sort_stats("tottime")
    #ps.print_stats("effective_energy_shift_optimization_BA")
    ps.print_stats()


def do_normal_mode(module, energy_excess_list, energy_deficit_list, start_time_phases, repetition_count):

    runtimes_single = []
    module_results = []

    for i in range(repetition_count):
        start = time.perf_counter()
        result_dict = module.process_phases(energy_excess_list[i], energy_deficit_list[i], start_time_phases)
        end = time.perf_counter()

        runtimes_single.append(end - start)
        module_results.append(result_dict)

    median_runtime = np.median(runtimes_single)

    output_runtime(module, median_runtime, repetition_count)

    return module_results, median_runtime

def execution_and_analysis(
        modules: list, energy_excess_list, energy_deficit_list, start_time_phases, repetition_count: int, submethod_analysis: bool, fake_run:bool):

    all_result_lists = []
    runtimes = []

    if fake_run:
        for m in modules:
            do_normal_mode(m, energy_excess_list, energy_deficit_list, start_time_phases, repetition_count=1)
        return

    for m in modules:
        if submethod_analysis:
            do_submethod_analysis(m, energy_excess_list, energy_deficit_list, start_time_phases)
            break
        else:
            module_results, median_runtime = do_normal_mode(m, energy_excess_list, energy_deficit_list, start_time_phases, repetition_count)
            all_result_lists.append(module_results)
            runtimes.append(median_runtime)

    return all_result_lists, runtimes


def has_program_run_long_enough(start_time, phase_count, time_limit):
    if time_limit is None:
        return False

    elapsed = time.perf_counter() - start_time

    if elapsed >= time_limit:
        elapsed_hours, rem = divmod(elapsed, 3600)
        elapsed_minutes, elapsed_seconds = divmod(rem, 60)
        print(delim)

        print(
            f"Time limit reached before starting phase_count = {phase_count} "
            f"(elapsed {int(elapsed_hours)}h {int(elapsed_minutes)}m {int(elapsed_seconds)}s). Stopping."
        )
        return True
    return False


def main(phase_counts: list, versions: list, indices: list, submethod_analysis: bool, repetition_count: int,
         master_seed, time_limit, worst_case_scenario, phase_count_for_submethod_analysis=20000):
    modules = get_modules(indices, versions)
    results = []

    start_time = time.perf_counter()

    # One Fake Run to compile all versions if needed
    energy_excess_list, energy_deficit_list, start_time_phases = init(worst_case_scenario, master_seed, 1, repetition_count)
    execution_and_analysis(modules, energy_excess_list, energy_deficit_list, start_time_phases, 1, submethod_analysis, True)

    for phase_count in phase_counts:

        if not submethod_analysis:
            print(delim)
            print("Current Phase Count: ", phase_count)

        if has_program_run_long_enough(start_time, phase_count, time_limit):
            break

        if submethod_analysis:

            energy_excess_list, energy_deficit_list, start_time_phases = init(worst_case_scenario, master_seed, phase_count_for_submethod_analysis, repetition_count)

            execution_and_analysis(modules, energy_excess_list, energy_deficit_list, start_time_phases, 1, submethod_analysis, False)
            break

        else:

            energy_excess_list, energy_deficit_list, start_time_phases = init(worst_case_scenario, master_seed, phase_count, repetition_count)

            result_dicts, runtimes = (execution_and_analysis
                                      (modules, energy_excess_list, energy_deficit_list, start_time_phases, repetition_count, submethod_analysis, False))

            test_result(result_dicts)
            results.append((phase_count, runtimes))

    return results


def phase_counts_generator(start: int, end: int, number_of_data_points: int):
    phase_counts = np.logspace(np.log10(start), np.log10(end), num=number_of_data_points)
    phase_counts = [int(x) for x in phase_counts]
    return sorted(list(set(phase_counts)))

def calculate_time_limit(time_limit_hours, time_limit_minutes, time_limit_seconds):
    time_limit = time_limit_hours * 3600 + time_limit_minutes * 60 + time_limit_seconds

    if time_limit_hours == 0 and time_limit_minutes == 0 and time_limit_seconds == 0:
        time_limit = None

    return time_limit


if __name__ == '__main__':

    cfg = load_config()

    create_result_folders_and_init_json(cfg)
    sys.exit()

    if cfg["only_data_fitting"]:
        log_log_linear_regression(cfg)
        sys.exit()

    phase_counts = phase_counts_generator(
        cfg["start_phase_count"],
        cfg["end_phase_count"],
        cfg["number_of_data_points"]
    )
    time_limit = calculate_time_limit(
        cfg["time_limit_hours"],
        cfg["time_limit_minutes"],
        cfg["time_limit_seconds"]
    )

    results = main(
        phase_counts,
        cfg["versions"],
        cfg["indices"],
        cfg["submethod_analysis"],
        cfg["repetition_count"],
        cfg["master_seed"],
        time_limit,
        cfg["worst_case_scenario"]
    )

    if not cfg["submethod_analysis"]:

        if not len(cfg["indices_to_save"]) == 0:
            save_to_json(cfg, results)

        plot_current_run(cfg, results)

        if not len(cfg["indices_to_save"]) == 0:
            plot_from_json(cfg, cfg["end_phase_count"])

        #log_log_linear_regression(cfg)

        plt.show()