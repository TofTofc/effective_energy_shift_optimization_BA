import json
import os

from matplotlib import pyplot as plt
import matplotlib.ticker as mticker


def plot_from_json(cfg,  max_phase_count=None):

    filename = "saved_data/worst_case.json" if cfg["worst_case_scenario"] else "saved_data/average_case.json"

    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} does not exist")

    with open(filename, "r") as f:
        json_data = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))

    for version_name, data in json_data.items():
        phase_counts = []
        runtimes = []
        for entry in data["results"]:
            phase_count = int(entry["phase_count"])
            runtime = float(entry["runtime"])
            if max_phase_count is None or phase_count <= max_phase_count:
                phase_counts.append(phase_count)
                runtimes.append(runtime)

        if phase_counts:
            ax.plot(phase_counts, runtimes, label=version_name)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.set_xlabel("Phase Count (log scale)")
    ax.set_ylabel("Runtime (s, log scale)")
    ax.set_title("Runtimes per Module vs Phase Count (From JSON Saved Data)")
    ax.legend()
    ax.grid(True, which="both", ls="--")

    if cfg["worst_case_scenario"]:

        os.makedirs("visuals_worst_case", exist_ok=True)

        fig.savefig("visuals_worst_case/Runtimes_from_json.png", dpi=300, bbox_inches="tight")

        if cfg["save_copy_of_visuals"]:
            fig.savefig("visuals_worst_case/Runtimes_from_json_copy.png", dpi=300, bbox_inches="tight")

    else:
        os.makedirs("visuals_average_case", exist_ok=True)

        fig.savefig("visuals_average_case/Runtimes_from_json.png", dpi=300, bbox_inches="tight")

        if cfg["save_copy_of_visuals"]:
            fig.savefig("visuals_average_case/Runtimes_from_json_copy.png", dpi=300, bbox_inches="tight")



def plot_current_run(cfg, results):
    version_to_index = {cfg["versions"][idx]: pos for pos, idx in enumerate(cfg["indices"])}

    fig, ax = plt.subplots(figsize=(10, 6))
    x_labels = [phase_count for phase_count, _ in results]
    x_vals = [float(x) for x in x_labels]

    for version_name in cfg["versions"]:
        if version_name in version_to_index:
            pos = version_to_index[version_name]
            y = [runtimes[pos] for _, runtimes in results]
            ax.plot(x_vals, y, label=version_name)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))

    ax.set_xlabel("Phase Count (log scale)")
    ax.set_ylabel("Runtime (s, log scale)")
    ax.set_title("Runtimes per Module vs Phase Count (From current run)")
    ax.legend()
    ax.grid(True, which="both", ls="--")

    if cfg["worst_case_scenario"]:

        os.makedirs("visuals_worst_case", exist_ok=True)

        fig.savefig("visuals_worst_case/Runtimes_current_run.png", dpi=300, bbox_inches="tight")

        if cfg["save_copy_of_visuals"]:
            fig.savefig("visuals_worst_case/Runtimes_current_run_copy.png", dpi=300, bbox_inches="tight")

    else:
        os.makedirs("visuals_average_case", exist_ok=True)

        fig.savefig("visuals_average_case/Runtimes_current_run.png", dpi=300, bbox_inches="tight")

        if cfg["save_copy_of_visuals"]:
            fig.savefig("visuals_average_case/Runtimes_current_run_copy.png", dpi=300, bbox_inches="tight")

