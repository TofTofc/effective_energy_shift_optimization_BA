import cProfile
import importlib
import importlib.util
import os
import pstats
import sys
import uuid
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

def dicts_equal(a: dict, b: dict) -> bool:
    for k in a.keys():
        v_a = a.get(k)
        v_b = b.get(k)

        #print(v_a)
        #print("-----------")
        #print(v_b)

        if not (v_a == v_b).all():
            return False
    return True


def test_result(dicts: list[dict]):
    first = dicts[0]
    for i, d in enumerate(dicts[1:], start=1):
        if not dicts_equal(first, d):
            sys.exit(f"Dictionaries at index 0 and {i} are not equal")
    #print("-------------------------------------------------------------------------------------------------------")
    #print("All results equal!")
    #print("-------------------------------------------------------------------------------------------------------")


def import_module(folder_name: str):
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, "versions", folder_name, "effective_energy_shift.py")

    module_name = f"effective_energy_shift_{folder_name}_{uuid.uuid4().hex}"

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def init(worst_case_scenario: bool, seed: int, phase_count: int):
    start_time_phases = np.arange(phase_count)

    if worst_case_scenario:
        energy_excess = np.arange(phase_count, 0, -1)
        energy_deficit = np.arange(1, phase_count + 1)

    else:
        rng = np.random.default_rng(seed)

        energy_excess = rng.integers(0, 10, phase_count)
        energy_deficit = rng.integers(0, 10, phase_count)

    return energy_excess, energy_deficit, start_time_phases


def get_modules(indices: list[int], versions: list):
    for i in indices:
        if i < 0 or i >= len(versions):
            sys.exit(f"Indices not suitable")

    modules = [import_module(versions[i]) for i in indices]
    return modules


def output_runtime(module, total_runtime: float):
    full_name = module.__name__
    short_name = full_name[len("effective_energy_shift_"):]
    short_name = "_".join(short_name.split("_")[:-1])

    print(f"Module: {short_name}, Avg runtime: {total_runtime / repetition_count:.8f}s")


def execution_and_analysis(modules: list, energy_excess, energy_deficit, start_time_phases,
                           repetition_count: int, submethod_analysis: bool):
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


def main(phase_counts: list, versions: list, indices: list, submethod_analysis: bool, repetition_count: int, seed, phase_count_for_submethod_analysis = 4000):
    modules = get_modules(indices, versions)
    results = []

    for phase_count in phase_counts:

        if not submethod_analysis:
            print("-------------------------------------------------------------------------------------------------------")
            print("Current Phase Count: ", phase_count)
            energy_excess, energy_deficit, start_time_phases = init(False, seed, phase_count)
            result_dicts, runtimes = execution_and_analysis(modules, energy_excess, energy_deficit, start_time_phases,repetition_count, submethod_analysis)
        else:
            print("Current Phase Count: ", phase_count_for_submethod_analysis)
            energy_excess, energy_deficit, start_time_phases = init(False, seed, phase_count_for_submethod_analysis)
            result_dicts, runtimes = execution_and_analysis(modules, energy_excess, energy_deficit, start_time_phases,1, submethod_analysis)
            test_result(result_dicts)
            break

        test_result(result_dicts)

        results.append((phase_count, runtimes))

    return results


def phase_counts_generator(start: int, end: int, factor: float):
    phase_counts = []
    value = start

    while value <= end:
        phase_counts.append(int(value))
        value *= factor

    return phase_counts


def plot(results, versions, indices):

    x_labels = [phase_count for phase_count, _ in results]
    x_pos = list(range(len(x_labels)))

    fig, ax = plt.subplots()

    for i, m in enumerate(indices):
        y = [runtimes[i] for _, runtimes in results]
        ax.plot(x_pos, y, marker='x', label=versions[m])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)

    ax.set_yscale('log')
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))

    ax.set_xlabel("Phase Count")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtimes per Module vs Phase Count")
    ax.legend()
    ax.grid(True, which="both", ls="--")
    plt.show()
    fig.savefig("Runtimes.png")

if __name__ == '__main__':
    start_count = 1
    end_count = 20000
    factor = 1.2

    phase_counts = phase_counts_generator(start_count, end_count, factor)

    versions = \
        [
            "original_simplified", "append_improved"
        ]
    indices = [0, 1]
    submethod_analysis = False
    repetition_count = 10
    seed = 123

    results = main(phase_counts, versions, indices, submethod_analysis, repetition_count, seed)
    if not submethod_analysis:
        plot(results, versions, indices)
