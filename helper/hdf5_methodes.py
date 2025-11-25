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
