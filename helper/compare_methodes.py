import sys

import numpy as np

delim = "-"*100

import numpy as np


def is_equal(tuple_a, tuple_b):
    starts_excess_list_a, starts_deficit_list_a, energy_excess_list_a, energy_deficit_list_a, mask_a = tuple_a
    starts_excess_list_b, starts_deficit_list_b, energy_excess_list_b, energy_deficit_list_b, mask_b = tuple_b

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

        if not np.array_equal(starts_a, starts_b):
            print(f"Versions are not equal at phase index i = {i}.")
            print("Deficit start points do not match.")
            return False

        if not np.array_equal(energy_a, energy_b):
            print(f"Versions are not equal at phase index i = {i}.")
            print("Deficit values (energy) do not match.")
            return False

    if not np.array_equal(mask_a[1], mask_b[1]):
        print("Error: Deficit Balance Masks (row 1) are not equal.")
        return False

    return True


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

            if current_start == last_end:

                merged_energy[-1] += current_energy
                last_end = merged_starts[-1] + merged_energy[-1]

            else:
                merged_starts.append(current_start)
                merged_energy.append(current_energy)
                last_end = current_start + current_energy

        merged_starts_list.append(np.array(merged_starts))
        merged_energy_list.append(np.array(merged_energy))

    return merged_starts_list, merged_energy_list