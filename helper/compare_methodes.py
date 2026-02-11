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
    print("::::::::")
    print(dict_a)
    print(dict_b)
    print("::::::::")

    phase_count = len(starts_excess_list_a)

    def check_invariants(d, label):
        cap = d["capacity"]
        eff = d["effectiveness_local"]

        # 1. Capacity at index 0 must be 0
        if not np.isclose(cap[0], 0, atol=1e-12):
            print(f"[{label}] Invariant Error: Initial capacity is not 0 (found {cap[0]})")
            return False

        # 2. Effectiveness: last element must be 0
        if not np.isclose(eff[-1], 0, atol=1e-12):
            print(f"[{label}] Invariant Error: Final effectiveness is not 0 (found {eff[-1]})")
            return False

        # 3. Effectiveness starts with phase_count
        if not np.isclose(eff[0], phase_count, atol=1e-12):
            print(f"[{label}] Invariant Error: Effectiveness starts with {eff[0]} instead of phase_count {phase_count}")
            return False

        # 4. Effectiveness must be monotonically decreasing (diff <= 0)
        if not np.all(np.diff(eff) <= 0):
            print(f"[{label}] Invariant Error: Effectiveness is not monotonically decreasing")
            return False

        return True

    # Validate both
    if not check_invariants(dict_a, "A") or not check_invariants(dict_b, "B"):
        return False

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

    eps = 10

    #capacity = np.unique(np.vectorize(round, otypes=[np.float64])(capacity_raw, eps))

    capacity_rounded = np.round(capacity_raw, decimals=eps)

    capacity_sorted = np.sort(capacity_rounded)

    if len(capacity_sorted) > 0:

        mask = np.ones(len(capacity_sorted), dtype=np.bool_)
        mask[1:] = np.diff(capacity_sorted) > 1e-12
        capacity = capacity_sorted[mask]
    else:
        capacity = capacity_sorted

    effectiveness_local = np.zeros(len(capacity))
    for start, energy_val in zip(capacity_phases, energy_additional_phases):
        upper_bound = start + energy_val
        effectiveness_local[(start <= capacity + 0.5 * 10**(-eps)) & (capacity < upper_bound - 0.5 * 10**(-eps))] += 1

    keep_mask = np.ones(len(effectiveness_local), dtype=bool)

    prev = -1

    for i in range(len(effectiveness_local)):

        if effectiveness_local[i] == prev:

            keep_mask[i] = False

        prev = effectiveness_local[i]

    capacity = capacity[keep_mask]
    effectiveness_local = effectiveness_local[keep_mask]

    return dict(capacity=capacity, effectiveness_local=effectiveness_local)
