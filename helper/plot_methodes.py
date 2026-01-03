import json
import os

from matplotlib import pyplot as plt
import matplotlib.ticker as mticker


def plot_from_json(cfg: dict, versions_to_plot=None, max_phase_count=None):

    case_type = "worst_case" if cfg.get("worst_case_scenario", False) else "average_case"

    base_folder = "results"
    runtimes_folder = os.path.join(base_folder, "runtimes")
    visuals_folder = os.path.join(base_folder, "visuals")

    average_folder = os.path.join(visuals_folder, "average_case")
    worst_folder = os.path.join(visuals_folder, "worst_case")
    copies_folder = os.path.join(visuals_folder, "copies")

    out_dir = average_folder if case_type == "average_case" else worst_folder

    if not versions_to_plot:
        version_folders = [f.path for f in os.scandir(runtimes_folder) if f.is_dir()]
    else:
        version_folders = [os.path.join(runtimes_folder, v) for v in versions_to_plot]

    fig, ax = plt.subplots(figsize=(10, 6))

    for version_folder in version_folders:
        json_file = os.path.join(version_folder, f"{case_type}.json")
        if not os.path.exists(json_file):
            continue
        with open(json_file, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        phase_counts = []
        runtimes = []
        for entry in json_data["results"]:
            phase_count = int(entry["phase_count"])
            runtime = float(entry["runtime"])
            if runtime > 0 and (max_phase_count is None or phase_count <= max_phase_count):
                phase_counts.append(phase_count)
                runtimes.append(runtime)

        if phase_counts:
            ax.plot(phase_counts, runtimes, label=json_data.get("version", os.path.basename(version_folder)))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.set_xlabel("Phase Count (log scale)")
    ax.set_ylabel("Runtime (s, log scale)")
    ax.set_title("Runtimes per Module vs Phase Count")
    ax.legend()
    ax.grid(True, which="both", ls="--")

    output_path = os.path.join(out_dir, "Runtimes_from_json.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if cfg.get("save_copy_of_visuals"):
        copy_path = os.path.join(copies_folder, "Runtimes_from_json_copy.png")
        fig.savefig(copy_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)
    return output_path