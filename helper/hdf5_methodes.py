import h5py
import numpy as np
from pathlib import Path

from helper.json_methodes import get_run_info_from_json


def save_simulation_results(all_data, phase_counts, cfg):

    print("saving data")

    version_name, _, repetition_count, master_seed, worst_case_scenario = get_run_info_from_json(cfg)

    output_dir = Path("results") / "output" / version_name
    output_dir.mkdir(parents=True, exist_ok=True)

    case_file = "worst_case.h5" if cfg.get("worst_case_scenario", False) else "average_case.h5"
    file_path = output_dir / case_file

    dt_int32_vlen = h5py.vlen_dtype(np.dtype('int32'))
    dt_int64_vlen = h5py.vlen_dtype(np.dtype('int64'))

    for cfg_idx, repetitions_list in enumerate(all_data):

        p_count = phase_counts[cfg_idx]

        with h5py.File(file_path, 'a') as f:

            f.attrs['version_name'] = version_name
            f.attrs['worst_case_scenario'] = worst_case_scenario
            f.attrs['repetition_count'] = repetition_count
            f.attrs['master_seed'] = master_seed

            cfg_group = f.create_group(f"phase_count_{p_count}")

            for rep_idx, data_arrays in enumerate(repetitions_list):

                starts_excess_list, starts_deficit_list, energy_excess_list, energy_deficit_list, mask = data_arrays

                rep_group = cfg_group.create_group(f"rep_{rep_idx}")

                collected_data = {
                    "starts_excess": starts_excess_list,
                    "starts_deficit": starts_deficit_list,
                    "energy_excess": energy_excess_list,
                    "energy_deficit": energy_deficit_list
                }

                for name, data_list in collected_data.items():

                    if "starts" in name:
                        dtype = dt_int64_vlen
                    else:
                        dtype = dt_int32_vlen

                    dset = rep_group.create_dataset(name, (len(data_list),), dtype=dtype, compression="gzip", compression_opts=4)
                    dset[:] = data_list

    print("saved data")


def vlen_array_equal(arr1, arr2) -> bool:
    if len(arr1) != len(arr2):
        return False

    for i in range(len(arr1)):
        if not np.array_equal(arr1[i], arr2[i]):
            print(arr1[i])
            print("_________________________________")
            print(arr2[i])
            print(i)
            return False

    return True