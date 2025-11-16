import numpy as np
from sklearn.linear_model import LinearRegression

delim = "-" * 100


def fit_loglog(n, t):
    n = np.array(n)
    t = np.array(t)

    mask = (n > 0) & (t > 0)
    n = n[mask]
    t = t[mask]

    log_n = np.log(n).reshape(-1, 1)
    log_t = np.log(t)

    model = LinearRegression().fit(log_n, log_t)

    p = model.coef_[0]
    a = np.exp(model.intercept_)

    t_pred = a * n ** p
    error = np.sum((t - t_pred) ** 2)

    return a, p, error


def fit_results_and_print(results):
    n_versions = len(results[0][1])
    n_list = [r[0] for r in results]

    for v in range(n_versions):
        t_list = [r[1][v] for r in results]
        a, p, error = fit_loglog(n_list, t_list)

        print(delim)
        print(f"Version {v}:")
        print(f"  t(n) ≈ {a:.3e} * n^{p:.3f}  (error={error:.3e})")
        print()
