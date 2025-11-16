import sys

import numpy as np


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


def compare_phase_objects(p1, p2):

    arrs1 = extract_phase_arrays(p1)
    arrs2 = extract_phase_arrays(p2)

    return all(np.array_equal(a1, a2) for a1, a2 in zip(arrs1, arrs2))


def dicts_equal(a: dict, b: dict) -> bool:

    if not np.array_equal(a['mask'], b['mask']):
        return False

    phases_a = a['phases']
    phases_b = b['phases']

    if len(phases_a) != len(phases_b):
        return False

    for p_a, p_b in zip(phases_a, phases_b):
        if not compare_phase_objects(p_a, p_b):
            return False

    return True


def test_result(all_result_lists: list[list[dict]]):

    repetition_count = len(all_result_lists[0])

    for i in range(repetition_count):
        first_dict = all_result_lists[0][i]
        for j, module_results in enumerate(all_result_lists[1:], start=1):
            if not dicts_equal(first_dict, module_results[i]):
                sys.exit(f"Dictionaries at repetition {i} of module 0 and module {j} are not equal")