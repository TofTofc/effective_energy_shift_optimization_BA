import numpy as np
from scipy.optimize import curve_fit
from math import factorial

delim = "-"*100

def f_const(x, a): return a
def f_n(x, a): return a * x
def f_n2(x, a): return a * x**2
def f_n3(x, a): return a * x**3
def f_log_n(x, a): return a * np.log(x)
def f_nlogn(x, a): return a * x * np.log(x)
def f_sqrt_n(x, a): return a * np.sqrt(x)
def f_fact_n(x, a): return a * np.array([factorial(int(xi)) for xi in x], dtype=float)
def f_n_0_75(x, a): return a * x**0.75
def f_n_1_5(x, a): return a * x**1.5
def f_n_2_5(x, a): return a * x**2.5

DEFAULT_CANDIDATES = [
    f_const, f_sqrt_n, f_log_n, f_n, f_nlogn,
    f_n2, f_n3, f_fact_n,
    f_n_0_75, f_n_1_5, f_n_2_5
]

FUNC_TO_O = {
    f_const: "O(1)",
    f_sqrt_n: "O(n^0.5)",
    f_log_n: "O(log n)",
    f_n: "O(n)",
    f_nlogn: "O(n log n)",
    f_n2: "O(n^2)",
    f_n3: "O(n^3)",
    f_fact_n: "O(n!)",
    f_n_0_75: "O(n^0.75)",
    f_n_1_5: "O(n^1.5)",
    f_n_2_5: "O(n^2.5)"
}

def fit_complexity(n, t, f_candidates=DEFAULT_CANDIDATES):
    results = []
    for f in f_candidates:
        try:
            popt, _ = curve_fit(f, n, t)
            residuals = t - f(n, *popt)
            error = np.sum(residuals**2)
            results.append((f, popt, error))
        except Exception:
            results.append((f, None, np.inf))
    results.sort(key=lambda x: x[2])
    return results

def fit_results_and_print(results, f_candidates=DEFAULT_CANDIDATES, top_k=3):
    n_versions = len(results[0][1])
    n_list = np.array([r[0] for r in results])

    for v in range(n_versions):
        t_list = np.array([r[1][v] for r in results])
        fits = fit_complexity(n_list, t_list, f_candidates)

        print(delim)
        print(f"Version {v}:")
        best = fits[0]
        print(f"  Most likely complexity: {FUNC_TO_O[best[0]]} (error={best[2]:.3e})")

        print(f"  Top {top_k} fits:")
        for f, param, error in fits[:top_k]:
            param_str = ", ".join(f"{p:.3e}" for p in param) if param is not None else "None"
            print(f"    {FUNC_TO_O[f]:<10} error={error:.3e}, param=[{param_str}]")
        print()
