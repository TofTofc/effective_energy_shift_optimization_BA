import importlib
import importlib.util
import os
import sys
import uuid
import time
import numpy as np

from helper.extract_results import extract_results
from helper.hdf5_methodes import save_simulation_results
from helper.json_methodes import save_to_json, load_config, get_run_info_from_json

delim = "-"*100
abort = False

def import_version(folder_name: str):

    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, "versions", folder_name, "effective_energy_shift.py")

    module_name = f"effective_energy_shift_{folder_name}_{uuid.uuid4().hex}"

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module


def init(worst_case_scenario: bool,
         master_seed: int,
         phase_count: int,
         repetition_count: int,
         guaranteed_balanced_phases: float = 0.1) \
         -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Initialize energy excess/deficit arrays and starting time of the phases."""

    start_time_phases = np.arange(phase_count)

    energy_excess_list = []
    energy_deficit_list = []

    num_balanced = int(guaranteed_balanced_phases * phase_count)

    for rep_index in range(repetition_count):

        if worst_case_scenario:
            energy_excess = np.arange(phase_count, 0, -1, dtype=np.float64)
            energy_deficit = np.arange(1, phase_count + 1, dtype=np.float64)
        else:

            # Unique for combination of master seed, phase count and repetition index
            # Different for Excess (added 0) and Deficit (added 1)
            ss_excess = np.random.SeedSequence([master_seed, phase_count, rep_index, 0])
            ss_deficit = np.random.SeedSequence([master_seed, phase_count, rep_index, 1])

            rng_excess = np.random.default_rng(ss_excess)
            rng_deficit = np.random.default_rng(ss_deficit)

            energy_excess = rng_excess.uniform(0, 100, phase_count)
            energy_deficit = rng_deficit.uniform(0, 100, phase_count)

        if num_balanced > 0:

            ss_balance = np.random.SeedSequence([master_seed, phase_count, rep_index, 2])
            rng_balance = np.random.default_rng(ss_balance)
            indices_to_balance = rng_balance.choice(phase_count, size=num_balanced, replace=False)
            energy_excess[indices_to_balance] = energy_deficit[indices_to_balance]

        energy_excess_list.append(energy_excess)
        energy_deficit_list.append(energy_deficit)

    return energy_excess_list, energy_deficit_list, start_time_phases


def output_runtime(module, total_runtime: float):
    full_name = module.__name__
    short_name = full_name[len("effective_energy_shift_"):]
    short_name = "_".join(short_name.split("_")[:-1])

    print(f"Module: {short_name}, Mean runtime: {total_runtime:.8f}s")


def do_normal_mode(module, energy_excess_list, energy_deficit_list, start_time_phases, save_to_hdf_till, repetition_count, fake_run):
    """Runs the given version repetition_count times and measures median runtime"""

    runtimes_single = []
    module_results = []

    for i in range(repetition_count):

        start = time.perf_counter()
        result = module.process_phases(energy_excess_list[i], energy_deficit_list[i], start_time_phases)
        end = time.perf_counter()

        runtimes_single.append(end - start)

        # Only save the results that we want to save
        if save_to_hdf_till >= len(energy_excess_list[i]):

            data_arrays = extract_results(result)
            module_results.append(data_arrays)

    median_runtime = np.median(runtimes_single)

    if not fake_run:
        output_runtime(module, median_runtime)

    return module_results, median_runtime


def main(save_to_hdf_till):

    cfg = load_config()

    phase_counts_done = []
    all_results = []

    saving_done = False

    while True:

        # Load needed info like repetition_count or current phase counts from the results folder
        version_name, pending_phase_counts, repetition_count, master_seed, worst_case_scenario = get_run_info_from_json(cfg)

        if not pending_phase_counts:
            print("Job done. Everything was measured")
            return

        module = import_version(version_name)

        # Fake run for numba compiling
        energy_excess_lists, energy_deficit_lists, start_time_phases = init(worst_case_scenario, master_seed, 10,repetition_count)
        do_normal_mode(module, energy_excess_lists, energy_deficit_lists, start_time_phases, save_to_hdf_till, repetition_count=1, fake_run=True)

        print(f"{delim}\n{delim}")
        print(f"Working on version: {version_name}, "f"phase counts from {pending_phase_counts[0]} to {pending_phase_counts[-1]}")
        print(delim)

        for phase_count in pending_phase_counts:

            cfg = load_config()
            if cfg["abort"]:
                if not saving_done:
                    save_simulation_results(all_results, phase_counts_done, cfg)
                return

            print(delim)
            print("Current Phase Count: ", phase_count)

            energy_excess_lists, energy_deficit_lists, start_time_phases = init(worst_case_scenario, master_seed, phase_count, repetition_count)

            module_results, median_runtime = do_normal_mode(module, energy_excess_lists, energy_deficit_lists, start_time_phases,save_to_hdf_till, repetition_count, fake_run=False)

            if not saving_done:

                all_results.append(module_results)
                phase_counts_done.append(phase_count)

                if phase_count >= save_to_hdf_till:
                    save_simulation_results(all_results, phase_counts_done, cfg)
                    all_results = []
                    phase_counts_done = []
                    saving_done = True

            save_to_json(cfg, phase_count, median_runtime, version_name)