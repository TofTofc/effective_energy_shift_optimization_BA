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
import json


def dicts_equal(a: dict, b: dict) -> bool:
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


def output_runtime(module, total_runtime: float, repetition_count):

    full_name = module.__name__
    short_name = full_name[len("effective_energy_shift_"):]
    short_name = "_".join(short_name.split("_")[:-1])

    print(f"Module: {short_name}, Avg runtime: {total_runtime / repetition_count:.8f}s")


def do_submethod_analysis(module, energy_excess, energy_deficit, start_time_phases, result_dicts):

    profile = cProfile.Profile()
    result_dict = profile.runcall(module.process_phases, energy_excess, energy_deficit, start_time_phases)
    result_dicts.append(result_dict)
    ps = pstats.Stats(profile).sort_stats("cumtime")
    ps.sort_stats("cumtime")
    ps.print_stats("effective_energy_shift_optimization_BA")


def do_normal_mode(module, energy_excess, energy_deficit, start_time_phases, result_dicts, runtimes, repetition_count):

    total_runtime = 0

    for i in range(repetition_count):

        start = time.perf_counter()
        result_dict = module.process_phases(energy_excess, energy_deficit, start_time_phases)
        end = time.perf_counter()

        total_runtime += end - start

        if i == 0:
            result_dicts.append(result_dict)

    runtimes.append(total_runtime / repetition_count)
    output_runtime(module, total_runtime, repetition_count)


def execution_and_analysis(
        modules: list, energy_excess, energy_deficit, start_time_phases,repetition_count: int, submethod_analysis: bool):

    result_dicts = []
    runtimes = []

    for m in modules:
        if submethod_analysis:
            do_submethod_analysis(m, energy_excess, energy_deficit, start_time_phases, result_dicts)
        else:
            do_normal_mode(m, energy_excess, energy_deficit, start_time_phases, result_dicts, runtimes,
                           repetition_count)

    return result_dicts, runtimes


def main(phase_counts: list, versions: list, indices: list, submethod_analysis: bool, repetition_count: int,
         seed, time_limit, phase_count_for_submethod_analysis=4000):

    modules = get_modules(indices, versions)
    results = []

    start_time = time.perf_counter()

    for phase_count in phase_counts:

        elapsed = -2
        #Time Limit enabled
        if not time_limit == -1:
            elapsed = time.perf_counter() - start_time

        if elapsed >= time_limit:
            elapsed_hours, rem = divmod(elapsed, 3600)
            elapsed_minutes, elapsed_seconds = divmod(rem, 60)
            print(
                "-------------------------------------------------------------------------------------------------------")
            print(
                f"Time limit reached before starting phase_count={phase_count} "
                f"(elapsed {int(elapsed_hours)}h {int(elapsed_minutes)}m {int(elapsed_seconds)}s). Stopping."
            )
            break

        if not submethod_analysis:
            print(
                "-------------------------------------------------------------------------------------------------------")
            print("Current Phase Count: ", phase_count)
            energy_excess, energy_deficit, start_time_phases = init(False, seed, phase_count)
            result_dicts, runtimes = execution_and_analysis(modules, energy_excess, energy_deficit, start_time_phases,
                                                            repetition_count, submethod_analysis)
        else:
            print("Current Phase Count: ", phase_count_for_submethod_analysis)
            energy_excess, energy_deficit, start_time_phases = init(False, seed, phase_count_for_submethod_analysis)
            result_dicts, runtimes = execution_and_analysis(modules, energy_excess, energy_deficit, start_time_phases,
                                                            1, submethod_analysis)
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
    x_vals = [float(x) for x in x_labels]

    fig, ax = plt.subplots()

    for i, m in enumerate(indices):
        y = [runtimes[i] for _, runtimes in results]
        ax.plot(x_vals, y, label=versions[m])

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))

    ax.set_xlabel("Phase Count (log scale)")
    ax.set_ylabel("Runtime (s, log scale)")
    ax.set_title("Runtimes per Module vs Phase Count")
    ax.legend()
    ax.grid(True, which="both", ls="--")

    fig.savefig("Runtimes.png", dpi=300, bbox_inches="tight")
    plt.show()


def load_config(filename="setup.json"):
    path = Path(__file__).parent / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_time_limit(time_limit_hours, time_limit_minutes, time_limit_seconds):
    time_limit = time_limit_hours * 3600 + time_limit_minutes * 60 + time_limit_seconds

    if time_limit_hours == 0 and time_limit_minutes == 0 and time_limit_seconds == 0:
        time_limit = -1

    return time_limit


if __name__ == '__main__':

    cfg = load_config()

    phase_counts = phase_counts_generator(
        cfg["start_phase_count"],
        cfg["end_phase_count"],
        cfg["growth_factor"]
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
        cfg["seed"],
        time_limit
    )

    if not cfg["submethod_analysis"]:
        plot(results, cfg["versions"], cfg["indices"])
