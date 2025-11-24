import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
delim = "-"*100

def log_log_linear_regression(cfg, version_name: str = None, max_phase_count=None):

    runtimes_folder = os.path.join("results", "runtimes")

    if version_name is not None:
        version_folders = [os.path.join(runtimes_folder, version_name)]
    else:
        version_folders = [f.path for f in os.scandir(runtimes_folder) if f.is_dir()]

    case_file = "worst_case.json" if cfg.get("worst_case_scenario", False) else "average_case.json"

    for vfolder in version_folders:
        json_path = os.path.join(vfolder, case_file)

        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        version_label = json_data.get("version", os.path.basename(vfolder))
        entries = json_data.get("results", [])
        phase_counts = []
        runtimes = []
        for entry in entries:
            phase_count = int(entry["phase_count"])
            runtime = float(entry["runtime"])
            if (max_phase_count is None) or (phase_count <= max_phase_count):
                phase_counts.append(phase_count)
                runtimes.append(runtime)

        x = np.array(phase_counts)
        y = np.array(runtimes)

        print(delim)
        print(f"Version: {version_label}")

        x_log = np.log10(x)
        y_log = np.log10(y)

        deg = 1
        p = np.polyfit(x_log, y_log, deg)
        b = p[0]
        log_a = p[1]
        a = 10 ** log_a

        print("log10(y) = log10(a) + b*log10(x)")
        print(f"log10(y) = {log_a:.5f} + {b:.5f} * log10(x)")
        print(f"y = {a:.5e} * x^{b:.5f}")
        print(delim)

        calculate_and_print_regression_stats(x_log, y_log, p)
        plot_powerlaw_fit(x, y, a, b, version_label)


def calculate_and_print_regression_stats(x_log, y_log, p):

    n = y_log.size
    m = p.size
    dof = n - m

    alpha = 0.05
    t_critical = stats.t.ppf(1 - alpha/2, dof)

    model = np.poly1d(p)
    y_model = model(x_log)

    y_bar = np.mean(y_log)
    R2 = np.sum((y_model - y_bar)**2) / np.sum((y_log - y_bar) ** 2)

    resid = y_log - y_model

    chi2 = np.sum((resid / y_model)**2)
    chi2_red = chi2 / dof

    std_err_log = np.sqrt(np.sum(resid ** 2) / dof)
    std_err_orig = 10 ** std_err_log

    print("Statistics")
    print(delim)
    print(f"  Degrees of freedom:                                    {dof}")
    print(f"  Critical t-value:                                     {t_critical: .4f}")
    print(f"  Coefficient of determination, R²:                     {R2: .4f}")
    print(f"  Chi-squared:                                          {chi2: .6f}")
    print(f"  Reduced chi-squared:                                  {chi2_red: .6f}")
    print(f"  Standard deviation of the error:                      {std_err_orig: .6f}")

def plot_powerlaw_fit(x, y, a, b, version_name):

    fig, ax = plt.subplots(figsize=(7,5))

    ax.scatter(x, y, color='gray', edgecolors='k', s=20, label='Data')

    x_fit = np.linspace(np.min(x), np.max(x), 100)
    y_fit = a * x_fit**b
    ax.plot(x_fit, y_fit, color='blue', lw=2, label=f'Fit: y = {a:.3e} * x^{b:.3f}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))

    ax.set_xlabel("Phase Count (log scale)")
    ax.set_ylabel("Runtime (s, log scale)")
    ax.set_title(f"Runtimes vs Phase Count - Version {version_name}")
    ax.legend()
    ax.grid(True, which="both", ls="--")

    plt.show()


