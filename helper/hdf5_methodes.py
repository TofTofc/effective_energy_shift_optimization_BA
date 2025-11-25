import h5py
import numpy as np
from pathlib import Path

from helper.compare_methodes import extract_phase_arrays
from helper.json_methodes import get_run_info_from_json


def save_simulation_results(all_data, phase_counts, cfg):

    version_name, _, repetition_count, master_seed, worst_case_scenario = get_run_info_from_json(cfg)

    output_dir = Path("results") / "output" / version_name
    output_dir.mkdir(parents=True, exist_ok=True)

    case_file = "worst_case.h5" if cfg.get("worst_case_scenario", False) else "average_case.h5"

    file_path = output_dir / case_file

    attr_names = [
        "starts_excess", "starts_deficit", "energy_excess", "energy_deficit",
        "excess_balanced", "deficit_balanced", "excess_ids"
    ]
    dt_float_vlen = h5py.vlen_dtype(np.dtype('float64'))
    dt_int_vlen = h5py.vlen_dtype(np.dtype('int64'))

    with h5py.File(file_path, 'a') as f:

        f.attrs['version_name'] = version_name
        f.attrs['worst_case_scenario'] = worst_case_scenario
        f.attrs['repetition_count'] = repetition_count
        f.attrs['master_seed'] = master_seed

        for cfg_idx, repetitions_list in enumerate(all_data):

            p_count = phase_counts[cfg_idx]
            cfg_group = f.create_group(f"phase_count_{p_count}")

            for rep_idx, result_dict in enumerate(repetitions_list):
                rep_group = cfg_group.create_group(f"rep_{rep_idx}")

                rep_group.create_dataset("mask", data=result_dict['mask'], compression="gzip", compression_opts=4)

                phases = result_dict['phases']
                collected_data = {name: [] for name in attr_names}

                for phase_obj in phases:
                    extracted = extract_phase_arrays(phase_obj)
                    for i, name in enumerate(attr_names):
                        collected_data[name].append(np.asarray(extracted[i]))

                for name, data_list in collected_data.items():
                    dtype = dt_int_vlen if "ids" in name else dt_float_vlen
                    dset = rep_group.create_dataset(
                        name, (len(phases),), dtype=dtype, compression="gzip", compression_opts=4
                    )
                    dset[:] = data_list

def vlen_array_equal(arr1, arr2) -> bool:
    if len(arr1) != len(arr2):
        return False

    for i in range(len(arr1)):
        if not np.array_equal(arr1[i], arr2[i]):
            return False

    return True

def compare_simulation_results(version_name_a,version_name_b, cfg) -> bool:

    case_file = "worst_case.h5" if cfg.get("worst_case_scenario") else "average_case.h5"

    path_a = Path("results") / "output" / version_name_a / case_file
    path_b = Path("results") / "output" / version_name_b / case_file

    if not path_a.exists():
        print(f" Error: No File for version: {version_name_a}")
        return False
    if not path_b.exists():
        print(f" Error: No File for version: {version_name_b}")
        return False

    print(f" Compare {version_name_a} with {version_name_b} (worst case scenario: {cfg.get("worst_case_scenario")})")

    with h5py.File(path_a, 'r') as f_a, h5py.File(path_b, 'r') as f_b:

        # Datasets, die verglichen werden müssen (Basierend auf Ihrer save-Funktion)
        # 'mask' muss hier eventuell auch rein, falls sie nicht im Loop ist
        datasets_to_compare = [
            "starts_excess", "starts_deficit", "energy_excess", "energy_deficit",
            "excess_balanced", "deficit_balanced", "excess_ids", "mask"
        ]

        groups_a = set(f_a.keys())
        groups_b = set(f_b.keys())

        # Bestimme die Schnittmenge der Phase Counts, die in beiden Dateien existieren
        common_phase_counts = sorted(list(groups_a.intersection(groups_b)))

        for cfg_group_name in common_phase_counts:
            cfg_group_a = f_a[cfg_group_name]
            cfg_group_b = f_b[cfg_group_name]

            reps_a = sorted(list(cfg_group_a.keys()))
            reps_b = sorted(list(cfg_group_b.keys()))

            if reps_a != reps_b:
                continue

            for rep_group_name in reps_a:
                rep_group_a = cfg_group_a[rep_group_name]
                rep_group_b = cfg_group_b[rep_group_name]

                for dset_name in datasets_to_compare:

                    data_a = rep_group_a[dset_name][:]
                    data_b = rep_group_b[dset_name][:]

                    if data_a.shape != data_b.shape:
                        return False

                    if not vlen_array_equal(data_a, data_b):
                        print(f"{dset_name} in {cfg_group_name}/{rep_group_name} is not equal")
                        return False

    print(f"Results for {version_name_a} and {version_name_b} are equal")
    return True