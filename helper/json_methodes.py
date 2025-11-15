import json
import os
import sys
from pathlib import Path

delim = ("-------------------------------------------------------------------------------------------------------")


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
