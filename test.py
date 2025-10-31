import cProfile
import importlib
import importlib.util
import os
import pstats
import sys
import uuid
import time

import numpy as np

def dicts_equal(a: dict, b:dict) -> bool:
    for k in a.keys():
        v_a = a.get(k)
        v_b = b.get(k)
        if not (v_a == v_b).all():
            return False
    return True

def test_result(dicts: list[dict]):

    first = dicts[0]
    for i, d in enumerate(dicts[1:], start=1):
        if not dicts_equal(first, d):
            sys.exit(f"Dictionaries at index 0 and {i} are not equal")
    print("------------------------")
    print("All results equal!")
    print("------------------------")

def import_module(folder_name: str):

    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, "versions", folder_name, "effective_energy_shift.py")

    module_name = f"effective_energy_shift_{folder_name}_{uuid.uuid4().hex}"

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module

def init(worst_case_scenario: bool, seed: int, phase_count):

    start_time_phases = np.arange(phase_count)

    if worst_case_scenario:
        energy_excess = np.arange(phase_count, 0, -1)
        energy_deficit = np.arange(1, phase_count + 1)

    else:
        rng = np.random.default_rng(seed)

        energy_excess = rng.integers(0, 10, phase_count)
        energy_deficit = rng.integers(0, 10, phase_count)

    return energy_excess, energy_deficit, start_time_phases

def get_modules(indices: list[int], versions):

    for i in indices:
        if i < 0 or i >= len(versions):
            sys.exit(f"Indices not suitable")

    modules = [import_module(versions[i]) for i in indices]
    return modules

def output_runtime(module, total_runtime):
    full_name = module.__name__
    short_name = full_name[len("effective_energy_shift_"):]
    short_name = "_".join(short_name.split("_")[:-1])

    print(f"Module: {short_name}, Avg runtime: {total_runtime / repetition_count:.6f}s")
def execution_and_analysis(modules,energy_excess, energy_deficit, start_time_phases, repetition_count):

    result_dicts = []
    runtimes = []

    for m in modules:
        if submethod_analysis:
            profile = cProfile.Profile()
            result_dict = profile.runcall(m.process_phases, energy_excess, energy_deficit, start_time_phases)
            result_dicts.append(result_dict)
            ps = pstats.Stats(profile).sort_stats("cumtime")
            ps.sort_stats("cumtime")
            ps.print_stats("effective_energy_shift_optimization_BA")
            ps.print_stats()
        else:
            total_runtime = 0

            for i in range(repetition_count):

                start = time.perf_counter()
                result_dict = m.process_phases(energy_excess, energy_deficit, start_time_phases)
                end = time.perf_counter()

                total_runtime += end - start

                if i == 0:
                    result_dicts.append(result_dict)

            runtimes.append(total_runtime / repetition_count)
            output_runtime(m, total_runtime)

    return result_dicts, runtimes


def main(phase_count, versions, indices, submethod_analysis, repetition_count):

    energy_excess, energy_deficit, start_time_phases = init(False, 123456789, phase_count)

    modules = get_modules(indices, versions)

    result_dicts, runtimes = execution_and_analysis(modules,energy_excess, energy_deficit, start_time_phases, repetition_count)

    test_result(result_dicts)


if __name__ == '__main__':

    phase_count = 1000
    versions = \
        [
            "append_improved", "original", "original_simplified"
        ]
    indices = [0, 1, 2]
    submethod_analysis = False
    repetition_count = 10

    main(phase_count, versions, indices, submethod_analysis, repetition_count)