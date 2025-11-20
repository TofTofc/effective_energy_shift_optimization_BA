import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
delim = "-"*100

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


def log_log_linear_regression(cfg, max_phase_count=None):

    filename = "saved_data/worst_case.json" if cfg.get("worst_case_scenario") else "saved_data/average_case.json"

    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} does not exist")

    with open(filename, "r") as f:
        json_data = json.load(f)

    for version_name, data in json_data.items():
        entries = data.get("results", [])
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
        print(f"Version: {version_name}")

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

        plot_powerlaw_fit(x, y, a, b, version_name)

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


