import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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


def log_log_linear_regression(cfg, results):

    version_to_index = {cfg["versions"][idx]: pos for pos, idx in enumerate(cfg["indices"])}

    x_vals = np.array([float(phase_count) for phase_count, _ in results])

    xy_by_version = {}

    for version_name in cfg["versions"]:
        if version_name in version_to_index:
            pos = version_to_index[version_name]
            y_vals = np.array([runtimes[pos] for _, runtimes in results])
            xy_by_version[version_name] = (x_vals, y_vals)

    for version_name, (x, y) in xy_by_version.items():

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

        plot_powerlaw_fit(x, y, a, b, version_name)



