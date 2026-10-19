"""
K-fold cross-validation for the local-GP model: tunes k, overlap_tau, nu (and optionally n_neighbors).

Setup (one-time):
  1. Change MODEL_MODULE below to the name of your solution file (without .py).
  2. In your Model class add a `nu` attribute and use it in create_gp():
         self.nu = 1.5                                   # in __init__ (before self.create_gp())
         Matern(..., nu=self.nu)                         # in create_gp()

Run:  python cross_validate.py
Output: progress per configuration, a ranked table, and cv_results.csv.

Notes:
  - Configurations are evaluated sequentially on purpose: Model.fit_model already fits the local
    GPs in parallel (joblib, n_jobs=-1), so parallelizing here too would oversubscribe the CPU.
  - All configurations use identical folds, so their costs are directly comparable (paired).
"""
import contextlib
import csv
import importlib
import io
import itertools
import time
import warnings

import numpy as np
from sklearn.model_selection import KFold

MODEL_MODULE = "solution"          # <- name of your file containing Model, calculate_cost, get_city_area_data

# ----------------------------------------------------------------------------------------------
# Search space (grid). Start coarse, then refine around the best region.
# ----------------------------------------------------------------------------------------------
GRID = {
    "k": [10, 15, 20, 30],
    "overlap_tau": [1.5, 2.0, 3.0, 4.0],
    "nu": [1.5, 2.5],              # only 0.5, 1.5, 2.5 are fast in scikit-learn's Matern
    "n_neighbors": [3],            # try [3, 5]: tau is capped by how many nearest centers are considered
}
N_SPLITS = 5
SEED = 0
RESULTS_CSV = "cv_results.csv"

mod = importlib.import_module(MODEL_MODULE)
Model, calculate_cost, get_city_area_data = mod.Model, mod.calculate_cost, mod.get_city_area_data


def make_model(params: dict):
    """Create a Model with the given hyperparameters (rebuilds the GPs since k and nu changed)."""
    model = Model()
    model.k = params["k"]
    model.overlap_tau = params["overlap_tau"]
    model.nu = params["nu"]
    model.n_neighbors = params["n_neighbors"]
    model.create_gp()
    return model


def evaluate(params: dict, X: np.ndarray, y: np.ndarray, flags: np.ndarray):
    """Mean and standard error of the task cost over the K folds."""
    costs = []
    for tr, va in KFold(N_SPLITS, shuffle=True, random_state=SEED).split(X):
        model = make_model(params)
        # silence the model's prints and sklearn's convergence warnings while tuning
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit_model(X[tr], y[tr], flags[tr])
            pred, _, _ = model.predict_pollution(X[va], flags[va])
        costs.append(calculate_cost(y[va], pred, flags[va]))
    costs = np.asarray(costs)
    return costs.mean(), costs.std(ddof=1) / np.sqrt(len(costs))


def main():
    train_x = np.loadtxt("train_x.csv", delimiter=",", skiprows=1)
    train_y = np.loadtxt("train_y.csv", delimiter=",", skiprows=1)
    test_x = np.loadtxt("test_x.csv", delimiter=",", skiprows=1)
    X, flags, _, _ = get_city_area_data(train_x, test_x)   # CV replaces the 10% validation split
    y = train_y

    keys = list(GRID)
    configs = [dict(zip(keys, vals)) for vals in itertools.product(*GRID.values())]
    configs = [c for c in configs if c["n_neighbors"] <= c["k"]]
    print(f"{len(configs)} configurations x {N_SPLITS} folds")

    results = []
    for i, params in enumerate(configs, 1):
        t0 = time.time()
        mean, se = evaluate(params, X, y, flags)
        dt = time.time() - t0
        results.append({**params, "cost": mean, "se": se, "seconds": dt})
        print(f"[{i:3d}/{len(configs)}] cost={mean:9.3f} ± {se:7.3f}  ({dt:6.1f}s)  {params}", flush=True)

    # save everything, best first
    results.sort(key=lambda r: r["cost"])
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\nTop 10 by mean CV cost:")
    for r in results[:10]:
        p = {k: r[k] for k in keys}
        print(f"  cost={r['cost']:9.3f} ± {r['se']:7.3f}  ({r['seconds']:6.1f}s)  {p}")

    # one-standard-error rule: among configs statistically tied with the best, take the cheapest to fit
    best = results[0]
    tied = [r for r in results if r["cost"] <= best["cost"] + best["se"]]
    cheapest = min(tied, key=lambda r: r["seconds"])
    print(f"\nBest:                  { {k: best[k] for k in keys} }")
    print(f"Cheapest within 1 SE:  { {k: cheapest[k] for k in keys} }")
    print("Afterwards: set these in Model.__init__ and refit on train + validation data.")


if __name__ == "__main__":
    main()