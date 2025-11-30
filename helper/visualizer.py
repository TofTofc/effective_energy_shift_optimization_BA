from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt


def find_dataset_by_substr(group, substr):
    substr = substr.lower()
    for name in group:
        if substr in name.lower():
            return name
    raise KeyError(f"No dataset in group contains '{substr}'")


def load_field_as_list(group, key):
    dset_name = find_dataset_by_substr(group, key)
    d = group[dset_name]
    return [np.asarray(d[i]) for i in range(len(d))]


def plot_columns(
    excess_list,
    deficit_list,
    starts_excess_list,
    starts_deficit_list,
    out_path,
    figsize=(14, 7),
    max_cols=0,
    width_excess=3.0,
    width_deficit=3.0,
    white_gap_width=0.5,
    pair_spacing=10.0,
    show_x_labels=True,
    edge_width=1.0
):
    n = len(excess_list)
    if max_cols > 0:
        n = min(n, max_cols)
        excess_list = excess_list[:n]
        deficit_list = deficit_list[:n]
        starts_excess_list = starts_excess_list[:n]
        starts_deficit_list = starts_deficit_list[:n]

    indices = np.arange(n) * pair_spacing
    fig, ax = plt.subplots(figsize=figsize)

    for i in range(n):
        x_center = indices[i]

        starts_e = np.asarray(starts_excess_list[i])
        values_e = np.asarray(excess_list[i])
        for b_start, b_height in zip(starts_e, values_e):
            if b_height == 0:
                continue
            ax.bar(
                x_center - (white_gap_width + width_excess)/2,
                b_height,
                width=width_excess,
                bottom=b_start,
                color="#e57373",
                edgecolor="black",
                linewidth=edge_width
            )

        starts_d = np.asarray(starts_deficit_list[i])
        values_d = np.asarray(deficit_list[i])
        for b_start, b_height in zip(starts_d, values_d):
            if b_height == 0:
                continue
            ax.bar(
                x_center + (white_gap_width + width_deficit)/2,
                b_height,
                width=width_deficit,
                bottom=b_start,
                color="#64b5f6",
                edgecolor="black",
                linewidth=edge_width
            )

    ax.set_xlim(-pair_spacing, indices[-1] + pair_spacing)

    if show_x_labels:
        ax.set_xticks(indices)
        ax.set_xticklabels([str(i) for i in range(n)], rotation=0, ha="center")
    else:
        ax.set_xticks([])

    ax.set_xlabel("Phase Column")
    ax.set_ylabel("Energy")
    ax.set_title("Excess (red) vs Deficit (blue) results")
    red_patch = plt.Rectangle((0,0),1,1,color="#e57373")
    blue_patch = plt.Rectangle((0,0),1,1,color="#64b5f6")
    ax.legend(handles=[red_patch, blue_patch], labels=["Excess", "Deficit"])
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.show()
    plt.close(fig)


def load_phase_data(cfg, version_name, phase=None, case="average_case"):
    folder = Path("results/output") / version_name
    file_path = folder / f"{case}.h5"

    if not file_path.exists():
        raise FileNotFoundError(f"HDF5 file not found at: {file_path}")

    with h5py.File(file_path, "r") as hf:
        if phase is None:
            phase = list(hf.keys())[0]
        elif isinstance(phase, int):
            phase = f"phase_count_{phase}"

        phase_group = hf[phase]
        rep_name = list(phase_group.keys())[0]
        rep_group = phase_group[rep_name]

        excess = load_field_as_list(rep_group, "energy_excess")
        deficit = load_field_as_list(rep_group, "energy_deficit")
        starts_excess = load_field_as_list(rep_group, "starts_excess")
        starts_deficit = load_field_as_list(rep_group, "starts_deficit")

        return excess, deficit, starts_excess, starts_deficit


def visualize(
    cfg,
    version_name,
    phase=None,
    case="average_case",
    figsize=(14, 7),
    max_cols=10,
    width_excess=3.0,
    width_deficit=3.0,
    white_gap_width=0.5,
    pair_spacing=10.0,
    show_x_labels=True,
    edge_width=1.3
):
    excess, deficit, starts_excess, starts_deficit = load_phase_data(
        cfg, version_name, phase, case
    )

    base_folder = Path("results/visuals_output") / version_name
    base_folder.mkdir(parents=True, exist_ok=True)

    if isinstance(phase, int):
        pc = phase
    else:
        pc = int(str(phase).replace("phase_count_", "")) if phase is not None else 0

    out_filename = f"{case}_output_pc_{pc}.png"
    out_path = base_folder / out_filename

    plot_columns(
        excess, deficit, starts_excess, starts_deficit,
        out_path=out_path,
        figsize=figsize,
        max_cols=max_cols,
        width_excess=width_excess,
        width_deficit=width_deficit,
        white_gap_width=white_gap_width,
        pair_spacing=pair_spacing,
        show_x_labels=show_x_labels,
        edge_width=edge_width
    )
