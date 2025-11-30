import os
import sys
import json
from pathlib import Path
import numpy as np

delim = "-"*100

def phase_counts_generator(start: int, end: int, number_of_data_points: int):

    phase_counts = np.logspace(np.log10(start), np.log10(end), num=number_of_data_points)
    phase_counts = [int(x) for x in phase_counts]
    return sorted(list(set(phase_counts)))

def load_config(filename="setup.json"):

    path = Path(sys.modules['__main__'].__file__).parent / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_to_json(cfg, phase_count, median_runtime, version_name, results_folder="results"):

    case_file = "worst_case.json" if cfg.get("worst_case_scenario", False) else "average_case.json"

    subfolder = Path(results_folder) / "runtimes" / version_name
    json_path = subfolder / case_file

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    phase_count_to_index = {entry["phase_count"]: i for i, entry in enumerate(data["results"])}

    if phase_count in phase_count_to_index:
        entry_index = phase_count_to_index[phase_count]
        current_runtime = data["results"][entry_index]["runtime"]
        if current_runtime == -1:
            data["results"][entry_index]["runtime"] = median_runtime

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def init_results_folders(cfg: dict, parent_folder: str = "results"):
    parent = Path(parent_folder)
    parent.mkdir(parents=True, exist_ok=True)

    runtimes_folder = parent / "runtimes"
    visuals_folder = parent / "visuals"
    visuals_output_folder = parent / "visuals_output"
    worst_case_folder = visuals_folder / "worst_case"
    average_case_folder = visuals_folder / "average_case"
    output_folder = parent / "output"

    for folder in (runtimes_folder, visuals_folder, visuals_output_folder, output_folder, worst_case_folder, average_case_folder):
        folder.mkdir(parents=True, exist_ok=True)

    versions = cfg.get("versions", [])

    repetition_count = cfg["repetition_count"]
    master_seed = cfg["master_seed"]
    start_phase_count = cfg["start_phase_count"]
    end_phase_count = cfg["end_phase_count"]
    number_of_data_points = cfg["number_of_data_points"]

    phase_counts = phase_counts_generator(
        start_phase_count,
        end_phase_count,
        number_of_data_points
    )

    for version in versions:
        runtimes_subpath = runtimes_folder / version
        runtimes_subpath.mkdir(parents=True, exist_ok=True)

        hdf5_subpath = output_folder / version
        hdf5_subpath.mkdir(parents=True, exist_ok=True)

        for case_name, worst_flag in (("average_case.json", False), ("worst_case.json", True)):
            json_path = runtimes_subpath / case_name
            if not json_path.exists():
                meta = {
                    "version": version,
                    "worst_case": worst_flag,
                    "number_of_data_points": number_of_data_points,
                    "start_phase_count": start_phase_count,
                    "end_phase_count": end_phase_count,
                    "repetition_count": repetition_count,
                    "master_seed": master_seed,
                    "results": [
                        {"phase_count": int(pc), "runtime": -1}
                        for pc in phase_counts
                    ],
                }
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)

        (visuals_output_folder / version).mkdir(parents=True, exist_ok=True)

def get_run_info_from_json(cfg: dict):

    case_file = "worst_case.json" if cfg.get("worst_case_scenario", False) else "average_case.json"
    runtimes_folder = os.path.join("results", "runtimes")

    index_list = cfg.get("index_to_use")
    if len(index_list) != 0:
        idx = index_list[0]
        version_name = cfg["versions"][idx]
    else:
        version_name = None

    version_folders = []
    if version_name is None:
        for f in os.scandir(runtimes_folder):
            if f.is_dir():
                json_path = os.path.join(f.path, case_file)
                if os.path.exists(json_path):
                    with open(json_path, "r", encoding="utf-8") as f_json:
                        data = json.load(f_json)
                        if any(entry["runtime"] == -1 for entry in data.get("results", [])):
                            version_name = data.get("version", os.path.basename(f.path))
                            version_folders = [f.path]
                            break
    else:
        version_folders = [os.path.join(runtimes_folder, version_name)]

    json_path = os.path.join(version_folders[0], case_file)
    with open(json_path, "r", encoding="utf-8") as f_json:
        data = json.load(f_json)

    results_list = data.get("results")

    pending_phase_counts = []
    max_entries = 10
    count = 0

    for entry in results_list:
        if entry["runtime"] == -1:
            if count < max_entries:
                pending_phase_counts.append(int(entry["phase_count"]))
                count += 1
            elif count >= max_entries:
                break
    repetition_count = data.get("repetition_count")
    master_seed = data.get("master_seed")
    worst_case_scenario = data.get("worst_case")

    return version_name, pending_phase_counts, repetition_count, master_seed, worst_case_scenario


def change_cfg(key, new_value):
    file = Path("setup.json")

    with open(file, 'r') as f:
        data = json.load(f)

    data[key] = new_value

    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

