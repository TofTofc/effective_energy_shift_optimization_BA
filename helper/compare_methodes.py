import sys
import json

delim = "-"*100

import numpy as np

def is_equal(tuple_a, tuple_b):
    starts_excess_list_a, starts_deficit_list_a, energy_excess_list_a, energy_deficit_list_a, mask_a = tuple_a
    starts_excess_list_b, starts_deficit_list_b, energy_excess_list_b, energy_deficit_list_b, mask_b = tuple_b

    dict_a = compute_battery_arrays_from_data(
        starts_excess_list_a,
        starts_deficit_list_a,
        energy_excess_list_a,
        energy_deficit_list_a,
        mask_a
    )

    dict_b = compute_battery_arrays_from_data(
        starts_excess_list_b,
        starts_deficit_list_b,
        energy_excess_list_b,
        energy_deficit_list_b,
        mask_b
    )

    np.set_printoptions(precision=16, suppress=True, threshold=sys.maxsize)
    print("Dict A:")
    print(dict_a)

    print("\nDict B:")
    print(dict_b)

    if dict_a["capacity"].size != dict_b["capacity"].size:
        print(f"Mismatch in capacity array size: A has {dict_a['capacity'].size}, B has {dict_b['capacity'].size}.")
        return False

    if not np.allclose(dict_a["capacity"], dict_b["capacity"], atol=1e-3):
        diff_idx = np.where(~np.isclose(dict_a["capacity"], dict_b["capacity"], atol=1e-3))[0][0]
        print(f"Mismatch in capacity at index {diff_idx}:")
        print(f"  Capacity A: {dict_a['capacity'][diff_idx]:.12f}")
        print(f"  Capacity B: {dict_b['capacity'][diff_idx]:.12f}")
        return False

    if dict_a["effectiveness_local"].size != dict_b["effectiveness_local"].size:
        print(f"Mismatch in effectiveness_local size: A has {dict_a['effectiveness_local'].size}, B has {dict_b['effectiveness_local'].size}.")
        return False

    if not np.allclose(dict_a["effectiveness_local"], dict_b["effectiveness_local"], atol=1e-3):
        diff_idx = np.where(~np.isclose(dict_a["effectiveness_local"], dict_b["effectiveness_local"], atol=1e-3))[0][0]
        print(f"Mismatch in effectiveness_local at index {diff_idx}:")
        print(f"  Effectiveness A: {dict_a['effectiveness_local'][diff_idx]:.12f}")
        print(f"  Effectiveness B: {dict_b['effectiveness_local'][diff_idx]:.12f}")
        return False

    return True


"""
starts_deficit_merged_a, energy_deficit_merged_a = merge_deficits(starts_deficit_list_a, energy_deficit_list_a)
starts_deficit_merged_b, energy_deficit_merged_b = merge_deficits(starts_deficit_list_b, energy_deficit_list_b)

for i in range(len(starts_deficit_merged_a)):

    starts_a = starts_deficit_merged_a[i]
    energy_a = energy_deficit_merged_a[i]
    starts_b = starts_deficit_merged_b[i]
    energy_b = energy_deficit_merged_b[i]

    if len(starts_a) != len(starts_b):
        print(f"Versions are not equal at phase index i = {i}.")
        print(f"Number of deficits is different: A has {len(starts_a)}, B has {len(starts_b)}.")
        return False

    if not np.allclose(starts_a, starts_b, atol=1e-3):

        diff_idx = np.where(~np.isclose(starts_a, starts_b, atol=1e-3))[0][0]
        print(f"Mismatch at phase {i}, Deficit index {diff_idx}:")
        print(f"  Start A: {starts_a[diff_idx]:.12f}")
        print(f"  Start B: {starts_b[diff_idx]:.12f}")
        return False

    if not np.allclose(energy_a, energy_b, atol=1e-3):

        diff_idx = np.where(~np.isclose(energy_a, energy_b, atol=1e-3))[0][0]
        print(f"Mismatch at phase {i}, Deficit index {diff_idx}:")
        print(f"  Energy A: {energy_a[diff_idx]:.12f}")
        print(f"  Energy B: {energy_b[diff_idx]:.12f}")
        return False

if not np.array_equal(mask_a[1], mask_b[1]):
    print("Error: Deficit Balance Masks (row 1) are not equal.")
    return False

return True
"""

def compute_battery_arrays_from_data(starts_excess_list, starts_deficit_list, energy_excess_list, energy_deficit_list, mask_list):

    capacity_phases = []
    energy_additional_phases = []

    deficit_masks = mask_list[1]

    for i in range(len(deficit_masks)):

        if not deficit_masks[i]:

            capacity_phases.extend(starts_deficit_list[i])
            energy_additional_phases.extend(energy_deficit_list[i])

        else:

            e_count = len(starts_excess_list[i])
            capacity_phases.extend(starts_deficit_list[i][:e_count])
            energy_additional_phases.extend(energy_deficit_list[i][:e_count])

    capacity_phases = np.array(capacity_phases)
    energy_additional_phases = np.array(energy_additional_phases)

    capacity = np.unique(np.sort(np.array([
        capacity_phases,
        capacity_phases + energy_additional_phases
    ]).flatten(), ))
    """
    capacity = np.unique(np.sort(np.round(np.array([
        capacity_phases,
        capacity_phases + energy_additional_phases
    ]).flatten(), 14)))
    """
    """
     capacity = np.unique(np.sort(np.round(np.array([
        capacity_phases,
        capacity_phases + energy_additional_phases
    ]).flatten(), 13)))
    """

    effectiveness_local = np.zeros(len(capacity))
    for start, energy_val in zip(capacity_phases, energy_additional_phases):
        upper_bound = start + energy_val
        effectiveness_local[(start <= capacity) & (capacity < upper_bound)] += 1

    keep_mask = np.ones(len(effectiveness_local), dtype=bool)

    keep_mask[1:] = np.diff(effectiveness_local) != 0

    capacity = capacity[keep_mask]
    effectiveness_local = effectiveness_local[keep_mask]

    return dict(capacity=capacity, effectiveness_local=effectiveness_local)




# Not used

def merge_deficits(starts_deficit_list, energy_deficit_list):
    merged_starts_list = []
    merged_energy_list = []

    for starts_arr, energy_arr in zip(starts_deficit_list, energy_deficit_list):

        merged_starts = [starts_arr[0]]
        merged_energy = [energy_arr[0]]
        last_end = starts_arr[0] + energy_arr[0]

        for i in range(1, len(starts_arr)):

            current_start = starts_arr[i]
            current_energy = energy_arr[i]

            if np.isclose(current_start, last_end, atol=1e-3):

                merged_energy[-1] += current_energy
                last_end = merged_starts[-1] + merged_energy[-1]

            else:
                merged_starts.append(current_start)
                merged_energy.append(current_energy)
                last_end = current_start + current_energy

        merged_starts_list.append(np.array(merged_starts))
        merged_energy_list.append(np.array(merged_energy))

    return merged_starts_list, merged_energy_list
