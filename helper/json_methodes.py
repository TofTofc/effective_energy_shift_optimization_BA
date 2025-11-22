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


def save_to_json(cfg, results):
    os.makedirs("../saved_data", exist_ok=True)
    filename = "saved_data/worst_case.json" if cfg["worst_case_scenario"] else "saved_data/average_case.json"

    try:
        with open(filename, "r") as f:
            output_json = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        output_json = {}

    for save_idx in cfg["indices_to_save"]:
        version_name = cfg["versions"][save_idx]
        output_json[version_name] = {
            "master_seed": cfg["master_seed"],
            "repetition_count": cfg["repetition_count"],
            "start_phase_count": cfg["start_phase_count"],
            "end_phase_count": cfg["end_phase_count"],
            "number_of_data_points": cfg["number_of_data_points"],
            "results": []
        }
        pos_in_results = cfg["indices"].index(save_idx)

        for phase_count, runtimes in results:
            runtime_value = float(runtimes[pos_in_results])
            output_json[version_name]["results"].append({
                "phase_count": int(phase_count),
                "runtime": float(runtime_value)
            })

    with open(filename, "w") as f:
        json.dump(output_json, f, indent=4)

    print(delim)
    print(f"Runtimes saved to {filename} for versions: {', '.join(cfg['versions'][i] for i in cfg['indices_to_save'])}")

def phase_counts_generator(start: int, end: int, number_of_data_points: int):
    phase_counts = np.logspace(np.log10(start), np.log10(end), num=number_of_data_points)
    phase_counts = [int(x) for x in phase_counts]
    return sorted(list(set(phase_counts)))

def create_version_folders_and_init_json(cfg: dict, parent_folder: str = "results"):

    parent = Path(parent_folder)
    parent.mkdir(parents=True, exist_ok=True)

    # Pflichtwerte aus cfg lesen und validieren
    versions = cfg.get("versions")
    if not versions:
        raise ValueError("cfg['versions'] darf nicht leer sein.")

    required_keys = ("repetition_count", "master_seed", "start_phase_count", "end_phase_count", "number_of_data_points")
    for key in required_keys:
        if cfg.get(key) is None:
            raise ValueError(f"cfg muss '{key}' enthalten.")

    repetition_count = int(cfg["repetition_count"])
    master_seed = int(cfg["master_seed"])
    start_phase_count = int(cfg["start_phase_count"])
    end_phase_count = int(cfg["end_phase_count"])
    number_of_data_points = int(cfg["number_of_data_points"])

    # Phase counts generieren (einmal)
    phase_counts = phase_counts_generator(start_phase_count, end_phase_count, number_of_data_points)

    # Basis-Metadaten
    base_meta = {
        "repetition_count": repetition_count,
        "master_seed": master_seed,
        "start_phase_count": start_phase_count,
        "end_phase_count": end_phase_count,
        "number_of_data_points": number_of_data_points,
    }

    # Suffix auswählen: worst_case oder average_case
    scenario_suffix = "_worst_case" if cfg.get("worst_case_scenario") else "_average_case"

    for version in versions:
        subfolder_name = f"{version}{scenario_suffix}"
        subpath = parent / subfolder_name

        # Ordner anlegen wenn nötig
        if not subpath.exists():
            subpath.mkdir(parents=True)
            created = True
        else:
            created = False

        # JSON-Datei heißt exakt wie der Subfolder (mit .json)
        json_filename = f"{subfolder_name}.json"
        json_path = subpath / json_filename

        # JSON nur erzeugen, wenn sie nicht existiert
        if not json_path.exists():
            meta = dict(base_meta)
            meta["version"] = version
            # results: alle phase_counts mit runtime = -1
            meta["results"] = [{"phase_count": int(pc), "runtime": -1} for pc in phase_counts]

            with json_path.open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            action = "created"
        else:
            action = "skipped (json exists)"

        print(f"{subpath} -> folder {'created' if created else 'exists'}, json {action}")

