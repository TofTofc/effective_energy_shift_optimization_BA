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

    if dict_a["capacity"].size != dict_b["capacity"].size:
        print(f"Mismatch in capacity array size: A has {dict_a['capacity'].size}, B has {dict_b['capacity'].size}.")
        return False

    if not np.allclose(dict_a["capacity"], dict_b["capacity"], atol=1e-12):
        diff_idx = np.where(~np.isclose(dict_a["capacity"], dict_b["capacity"], atol=1e-12))[0][0]
        print(f"Mismatch in capacity at index {diff_idx}:")
        print(f"  Capacity A: {dict_a['capacity'][diff_idx]:.12f}")
        print(f"  Capacity B: {dict_b['capacity'][diff_idx]:.12f}")
        return False

    if dict_a["effectiveness_local"].size != dict_b["effectiveness_local"].size:
        print(f"Mismatch in effectiveness_local size: A has {dict_a['effectiveness_local'].size}, B has {dict_b['effectiveness_local'].size}.")
        return False

    if not np.allclose(dict_a["effectiveness_local"], dict_b["effectiveness_local"], atol=1e-12):
        diff_idx = np.where(~np.isclose(dict_a["effectiveness_local"], dict_b["effectiveness_local"], atol=1e-12))[0][0]
        print(f"Mismatch in effectiveness_local at index {diff_idx}:")
        print(f"  Effectiveness A: {dict_a['effectiveness_local'][diff_idx]:.12f}")
        print(f"  Effectiveness B: {dict_b['effectiveness_local'][diff_idx]:.12f}")
        return False

    return True

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

    capacity_raw = np.sort(np.array([
        capacity_phases,
        capacity_phases + energy_additional_phases
    ]).flatten())

    capacity = np.unique(np.round(capacity_raw, 12))

    eps = 1e-12

    effectiveness_local = np.zeros(len(capacity))
    for start, energy_val in zip(capacity_phases, energy_additional_phases):
        upper_bound = start + energy_val
        effectiveness_local[(start <= capacity + eps) & (capacity < upper_bound - eps)] += 1

    keep_mask = np.ones(len(effectiveness_local), dtype=bool)

    prev = -1

    for i in range(len(effectiveness_local)):

        if effectiveness_local[i] == prev:

            keep_mask[i] = False

        prev = effectiveness_local[i]

    capacity = capacity[keep_mask]
    effectiveness_local = effectiveness_local[keep_mask]

    return dict(capacity=capacity, effectiveness_local=effectiveness_local)
