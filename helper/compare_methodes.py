import sys

import numpy as np

delim = "-"*100

def extract_phase_arrays(phase):

    expected_attrs = [
        "capacity_excess", "capacity_deficit", "size_excess", "size_deficit",
        "starts_excess", "starts_deficit",
        "energy_excess", "energy_deficit",
    ]

    all_exist = all(hasattr(phase, attr) for attr in expected_attrs)

    if all_exist:
        size_excess = getattr(phase, "size_excess")
        size_deficit = getattr(phase, "size_deficit")

        starts_excess = getattr(phase, "starts_excess")[:size_excess]
        starts_deficit = getattr(phase, "starts_deficit")[:size_deficit]
        energy_excess_arr = getattr(phase, "energy_excess")[:size_excess]
        energy_deficit_arr = getattr(phase, "energy_deficit")[:size_deficit]

    else:

        starts_excess = getattr(phase, "starts_excess")
        starts_deficit = getattr(phase, "starts_deficit")
        energy_excess_arr = getattr(phase, "energy_excess")
        energy_deficit_arr = getattr(phase, "energy_deficit")

    return (
        starts_excess,
        starts_deficit,
        energy_excess_arr,
        energy_deficit_arr,
    )


def compare_phase_objects(p1, p2):

    arrs1 = extract_phase_arrays(p1)
    arrs2 = extract_phase_arrays(p2)

    return all(np.array_equal(a1, a2) for a1, a2 in zip(arrs1, arrs2))



def repetitions_equal(rep_a: list, rep_b: list) -> bool:

    if len(rep_a) != len(rep_b):
        return False

    for p_a, p_b in zip(rep_a, rep_b):
        if not compare_phase_objects(p_a, p_b):
            return False
    return True


def test_results(list_a: list[list], list_b: list[list]):

    for i, (rep_a, rep_b) in enumerate(zip(list_a, list_b)):
        if not repetitions_equal(rep_a, rep_b):
            sys.exit(f"Repetition {i} not equal")

    print("Both results are equal!")
    return True


"""

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