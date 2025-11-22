import os
import sys
import json
from pathlib import Path
import numpy as np

delim = "-"*100


def load_config(filename="setup.json"):

    path = Path(sys.modules['__main__'].__file__).parent / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_to_json(cfg, results, results_folder="results"):

    case_file = "worst_case.json" if cfg.get("worst_case_scenario", False) else "average_case.json"

    for idx in cfg["indices"]:
        version_name = cfg["versions"][idx]
        subfolder = Path(results_folder) / version_name
        json_path = subfolder / case_file

        if not json_path.exists():
            raise FileNotFoundError(f"{json_path} existiert nicht!")

        # JSON laden
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Map phase_count -> index im JSON results
        phase_count_to_index = {entry["phase_count"]: i for i, entry in enumerate(data["results"])}

        # Für alle neuen Messungen den runtime-Wert aktualisieren, falls -1
        pos_in_results = cfg["indices"].index(idx)  # Position im results-Array
        for phase_count, runtimes_array in results:
            if phase_count not in phase_count_to_index:
                continue  # Phase_count existiert nicht in JSON, ignorieren
            entry_index = phase_count_to_index[phase_count]
            current_runtime = data["results"][entry_index]["runtime"]
            if current_runtime == -1:
                measured_runtime = runtimes_array[pos_in_results]
                data["results"][entry_index]["runtime"] = measured_runtime

        # JSON zurückschreiben
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)


def phase_counts_generator(start: int, end: int, number_of_data_points: int):
    phase_counts = np.logspace(np.log10(start), np.log10(end), num=number_of_data_points)
    phase_counts = [int(x) for x in phase_counts]
    return sorted(list(set(phase_counts)))

def create_result_folders_and_init_json(cfg: dict, parent_folder: str = "results"):

    parent = Path(parent_folder)
    parent.mkdir(parents=True, exist_ok=True)

    versions = cfg.get("versions")

    repetition_count = cfg["repetition_count"]
    master_seed = cfg["master_seed"]
    start_phase_count = cfg["start_phase_count"]
    end_phase_count = cfg["end_phase_count"]
    number_of_data_points = cfg["number_of_data_points"]

    phase_counts = phase_counts_generator(start_phase_count, end_phase_count, number_of_data_points)

    for version in versions:
        subpath = parent / version
        subpath.mkdir(parents=True, exist_ok=True)

        # Zwei JSON-Dateien erzeugen: average_case und worst_case
        for case_name, worst_flag in (("average_case.json", False), ("worst_case.json", True)):
            json_path = subpath / case_name
            if not json_path.exists():
                # Metadaten
                meta = {
                    "version": version,
                    "worst_case": worst_flag,
                    "number_of_data_points": number_of_data_points,
                    "start_phase_count": start_phase_count,
                    "end_phase_count": end_phase_count,
                    "repetition_count": repetition_count,
                    "master_seed": master_seed,
                    "results": [{"phase_count": int(pc), "runtime": -1} for pc in phase_counts]
                }

                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)

