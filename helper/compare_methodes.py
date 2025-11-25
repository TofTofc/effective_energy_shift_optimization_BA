import sys

import numpy as np

delim = "-"*100

def extract_phase_arrays(phase):

    expected_attrs = [
        "capacity_excess", "capacity_deficit", "size_excess", "size_deficit",
        "starts_excess", "starts_deficit",
        "energy_excess", "energy_deficit",
        "excess_balanced", "deficit_balanced",
        "excess_ids"
    ]

    all_exist = all(hasattr(phase, attr) for attr in expected_attrs)

    if all_exist:
        size_excess = getattr(phase, "size_excess")
        size_deficit = getattr(phase, "size_deficit")

        starts_excess = getattr(phase, "starts_excess")[:size_excess]
        starts_deficit = getattr(phase, "starts_deficit")[:size_deficit]
        energy_excess_arr = getattr(phase, "energy_excess")[:size_excess]
        energy_deficit_arr = getattr(phase, "energy_deficit")[:size_deficit]
        excess_balanced = getattr(phase, "excess_balanced")[:size_excess]
        deficit_balanced = getattr(phase, "deficit_balanced")[:size_deficit]
        excess_ids = getattr(phase, "excess_ids")[:size_excess]

    else:

        starts_excess = getattr(phase, "starts_excess")
        starts_deficit = getattr(phase, "starts_deficit")
        energy_excess_arr = getattr(phase, "energy_excess")
        energy_deficit_arr = getattr(phase, "energy_deficit")
        excess_balanced = getattr(phase, "excess_balanced")
        deficit_balanced = getattr(phase, "deficit_balanced")
        excess_ids = getattr(phase, "excess_ids")

    return (
        starts_excess,
        starts_deficit,
        energy_excess_arr,
        energy_deficit_arr,
        excess_balanced,
        deficit_balanced,
        excess_ids
    )

"""
def compare_phase_objects(p1, p2):

    arrs1 = extract_phase_arrays(p1)
    arrs2 = extract_phase_arrays(p2)

    print(arrs1)
    print(arrs2)
    print(delim)

    return all(np.array_equal(a1, a2) for a1, a2 in zip(arrs1, arrs2))


def dicts_equal(a: dict, b: dict) -> bool:

    #if not np.array_equal(a['mask'], b['mask']):
        #return False

    phases_a = a['phases']
    phases_b = b['phases']

    if len(phases_a) != len(phases_b):
        return False

    for p_a, p_b in zip(phases_a, phases_b):
        compare_phase_objects(p_a, p_b)
        #if not compare_phase_objects(p_a, p_b):
            #return False

    return True


def test_result(all_result_lists: list[list[dict]]):

    repetition_count = len(all_result_lists[0])

    for i in range(repetition_count):
        first_dict = all_result_lists[0][i]
        for j, module_results in enumerate(all_result_lists[1:], start=1):
            if not dicts_equal(first_dict, module_results[i]):
                sys.exit(f"Dictionaries at repetition {i} of module 0 and module {j} are not equal")


def test_result(all_result_dicts: list[dict]):

    for idx, result_dict in enumerate(all_result_dicts):

        phases = result_dict["phases"]

        battery_dict = compute_battery_arrays_from_phases(phases)

        print("\n--- Ergebnis für Dict", idx, "---")
        print("capacity:", battery_dict["capacity"])
        print("energy_additional:", battery_dict["energy_additional"])
        print("effectiveness_local:", battery_dict["effectiveness_local"])
        print("--------------------------------\n")

    sys.exit()

def compute_battery_arrays_from_phases(phases, efficiency_discharging = 1):

    capacity_phases = []
    energy_additional_phases = []

    for phase in phases:
        capacity_phases.extend(phase.starts_deficit[phase.deficit_balanced])
        energy_additional_phases.extend(phase.energy_deficit[phase.deficit_balanced])

    capacity_phases = np.array(capacity_phases)
    energy_additional_phases = np.array(energy_additional_phases)

    capacity = np.unique(np.sort(np.array([capacity_phases, capacity_phases + energy_additional_phases]).flatten()))

    effectiveness_local = np.zeros(len(capacity))
    for phase in phases:
        for capacity_lower, capacity_upper in zip(phase.starts_deficit[phase.deficit_balanced], phase.starts_deficit[phase.deficit_balanced] + phase.energy_deficit[phase.deficit_balanced]):
            effectiveness_local[(capacity_lower <= capacity) & (capacity < capacity_upper)] += 1

    delta_capacity = np.diff(capacity)
    delta_energy_additional = effectiveness_local[:-1]*delta_capacity
    energy_additional = efficiency_discharging * np.array([0, *delta_energy_additional.cumsum()])

    return dict(capacity=capacity, energy_additional=energy_additional, effectiveness_local=effectiveness_local)
    
"""