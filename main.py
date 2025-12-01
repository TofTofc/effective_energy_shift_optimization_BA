import importlib
import importlib.util
import os
import sys
import uuid
import time
import numpy as np

from helper.hdf5_methodes import save_simulation_results, compare_simulation_results
from helper.json_methodes import save_to_json, load_config, init_results_folders, get_run_info_from_json, change_cfg
from helper.plot_methodes import  plot_from_json
from helper.runtime_fitting_methodes import log_log_linear_regression

delim = "-"*100
abort = False

def import_version(folder_name: str):

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
    """Initialize energy excess/deficit arrays and starting time of the phases."""

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

            """ 
            # Make the total sums equal by distributing the difference
            sum_ex = energy_excess.sum()
            sum_def = energy_deficit.sum()
            diff = sum_ex - sum_def

            if diff > 0:
                q, r = divmod(diff, phase_count)
                if q:
                    energy_deficit = energy_deficit + q
                if r:
                    energy_deficit[:r] = energy_deficit[:r] + 1
            elif diff < 0:
                needed = -diff
                q, r = divmod(needed, phase_count)
                if q:
                    energy_excess = energy_excess + q
                if r:
                    energy_excess[:r] = energy_excess[:r] + 1
            """

        energy_excess_list.append(energy_excess)
        energy_deficit_list.append(energy_deficit)

    return energy_excess_list, energy_deficit_list, start_time_phases


def output_runtime(module, total_runtime: float, repetition_count):
    full_name = module.__name__
    short_name = full_name[len("effective_energy_shift_"):]
    short_name = "_".join(short_name.split("_")[:-1])

    print(f"Module: {short_name}, Mean runtime: {total_runtime:.8f}s")


def do_normal_mode(module, energy_excess_list, energy_deficit_list, start_time_phases, repetition_count, fake_run):
    """Runs the given version repetition_count times and measures median runtime"""

    global abort

    runtimes_single = []
    module_results = []

    for i in range(repetition_count):

        # Terminates the run
        cfg = load_config()
        if cfg["abort"]:
            abort = True

        start = time.perf_counter()
        phases_list = module.process_phases(energy_excess_list[i], energy_deficit_list[i], start_time_phases)
        end = time.perf_counter()

        runtimes_single.append(end - start)

        # Only save the results that we later also want to save
        # -> great memory savings
        if cfg["save_to_hdf5_till_count"] >= len(energy_excess_list[i]):
            module_results.append(phases_list)

    median_runtime = np.median(runtimes_single)

    if not fake_run:
        output_runtime(module, median_runtime, repetition_count)

    return module_results, median_runtime


def main():

    global abort

    cfg = load_config()

    phase_counts_done = []
    all_results = []

    while True:

        # Load needed info like repetition_count or current phase counts from the results folder
        version_name, pending_phase_counts, repetition_count, master_seed, worst_case_scenario = get_run_info_from_json(cfg)

        if not pending_phase_counts:
            print("Job done. Everything was measured")
            print("saving data")
            save_simulation_results(all_results, phase_counts_done, cfg)
            print("saved data")
            return

        module = import_version(version_name)

        # Fake run for numba compiling
        energy_excess_lists, energy_deficit_lists, start_time_phases = init(worst_case_scenario, master_seed, 10,repetition_count)
        do_normal_mode(module, energy_excess_lists, energy_deficit_lists, start_time_phases, repetition_count=1, fake_run=True)

        print(f"{delim}\n{delim}")
        print(f"Working on version: {version_name}, "f"phase counts from {pending_phase_counts[0]} to {pending_phase_counts[-1]}")
        print(delim)

        for phase_count in pending_phase_counts:

            print(delim)
            print("Current Phase Count: ", phase_count)

            energy_excess_lists, energy_deficit_lists, start_time_phases = init(worst_case_scenario, master_seed, phase_count, repetition_count)

            module_results, median_runtime = do_normal_mode(module, energy_excess_lists, energy_deficit_lists, start_time_phases, repetition_count, fake_run=False)

            if abort:
                print("saving data")
                save_simulation_results(all_results, phase_counts_done, cfg)
                print("saved data")
                return

            phase_counts_done.append(phase_count)
            all_results.append(module_results)

            save_to_json(cfg, phase_count, median_runtime, version_name)
    return