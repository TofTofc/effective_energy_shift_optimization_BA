import json
from pathlib import Path
import csv

def convert_jsons_to_csv(input_dir="results/runtimes", output_dir="results/csv"):

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    for json_file in input_dir.rglob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)

        rows = [(r["phase_count"], r["runtime"]) for r in data["results"] if r["runtime"] != -1]
        rel_path = json_file.relative_to(input_dir).with_suffix(".csv")
        out_file = output_dir / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(["phase_count", "runtime"])
            writer.writerows(rows)