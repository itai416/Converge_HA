"""Convergence diagnostics for the ablation models (run after scripts/06_ablation_ABC.py).

Three questions, answered on the primary split (repeat 0 = folds.csv) with the hyperparameters inner CV chose per fold:
1. Optimisation: does every Huber-ridge fit reach the minimum? Convex objective -> check the L-BFGS success flag, the
   iteration count and the final gradient size (max |gradient|, relative to its value at the starting point), for every
   configuration of every tuning grid on every outer training set.
2. Boosting length: where does held-out loss stop improving, and where did inner CV choose to stop? Curves are refit
   to 500 rounds; the dot marks the round chosen in that fold. Outer-fold curves are diagnostic only, never used to tune.
3. Grid edges: hyperparameters chosen at a grid boundary (from grid_edges.csv, written by the runner).
Outputs in results/ablation_ABC/: convergence.png, convergence_ridge_grid.csv
"""
import importlib
import itertools
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
ablation = importlib.import_module("06_ablation_ABC")

OUT = ROOT / "results" / "ablation_ABC"
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
MAX_ROUNDS = 500


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#bdbcb6")
    ax.tick_params(colors=MUTED, labelsize=8)


def lgbm_panel(ax, name, df, folds, exps, chosen):
    cls, fixed, _, X = exps[name]
    y = df["ddG"].values
    trains, tests, rows = [], [], []
    for k in range(5):
        tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
        p = dict(chosen[f"{name} | huber | none | rep0 | 0"][str(k)])
        picked = p.pop("n_estimators")
        m = cls(**fixed, loss="huber", seed=0, n_estimators=MAX_ROUNDS, **p).fit(
            X.iloc[tr], y[tr], eval_sets=[(X.iloc[tr], y[tr]), (X.iloc[te], y[te])])
        a, b = np.array(m.evals_result_["valid_0"]["huber"]), np.array(m.evals_result_["valid_1"]["huber"])
        trains.append(a), tests.append(b)
        ax.plot(a, color=BLUE, lw=0.8, alpha=0.3)
        ax.plot(b, color=ORANGE, lw=0.8, alpha=0.3)
        ax.plot(picked - 1, b[picked - 1], "o", ms=8, color=ORANGE, mec="white", mew=2, zorder=5)
        rows.append({"model": name, "fold": k, "chosen_round": picked, "heldout_best_round": int(b.argmin()) + 1,
                     "heldout_at_chosen": b[picked - 1], "heldout_best": b.min(), "heldout_at_500": b[-1]})
    ax.plot(np.mean(trains, 0), color=BLUE, lw=2, label="training rows (mean of 5 folds)")
    ax.plot(np.mean(tests, 0), color=ORANGE, lw=2, label="held-out fold (mean of 5 folds)")
    ax.plot([], [], "o", ms=8, color=ORANGE, mec="white", mew=2, ls="none", label="round chosen by inner CV")
    ax.set_title(name, fontsize=10, color=INK, loc="left")
    ax.set_xlabel("boosting round", fontsize=9, color=MUTED)
    ax.set_ylabel("Huber loss (δ = 1)", fontsize=9, color=MUTED)
    return rows


def _fit_ridge(cls, fixed, params, X, y, track):
    m = cls(**fixed, loss="huber", track_history=track, **params).fit(X, y)
    return m.history_, m.converged_, m.n_iter_, m.grad_norm_ / m.grad_norm0_


def ridge_panel(ax, name, df, folds, exps, chosen):
    cls, fixed, _, X = exps[name]
    y = df["ddG"].values
    for k in range(5):
        tr = np.where(folds != k)[0]
        h, _, _, _ = _fit_ridge(cls, fixed, chosen[f"{name} | huber | none | rep0 | None"][str(k)], X.iloc[tr], y[tr], True)
        h = np.array(h)
        ax.semilogy(np.arange(1, len(h) + 1), (h - h[-1]) / abs(h[-1]) + 1e-12, color=BLUE, lw=1.2, alpha=0.8,
                    label="one outer fold" if k == 0 else None)
    ax.set_title(name, fontsize=10, color=INK, loc="left")
    ax.set_xlabel("L-BFGS iteration", fontsize=9, color=MUTED)
    ax.set_ylabel("relative gap to final objective", fontsize=9, color=MUTED)


def ridge_grid_check(df, folds, exps):
    y = df["ddG"].values
    jobs = []
    for name, (cls, fixed, grid, X) in exps.items():
        if cls.__name__ != "AugmentedRidge":
            continue
        configs = [dict(zip(grid, v)) for v in itertools.product(*grid.values())] or [{}]
        for k in range(5):
            tr = np.where(folds != k)[0]
            for p in configs:
                jobs.append(((name, k, str(p)), (cls, fixed, p, X.iloc[tr], y[tr], False)))
    res = Parallel(n_jobs=15)(delayed(_fit_ridge)(*a) for _, a in jobs)
    return pd.DataFrame([{"model": n, "fold": k, "params": p, "converged": ok, "n_iter": it, "rel_grad": g}
                         for ((n, k, p), _), (_, ok, it, g) in zip(jobs, res)])


def main():
    _, df = ablation.load("35M")
    folds = df[["complex"]].merge(pd.read_csv(ROOT / "data" / "processed" / "folds.csv"), on="complex")[
        "fold_complex"].values
    exps = ablation.experiments(df)
    chosen = json.loads((OUT / "chosen_params.json").read_text())

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax in axes.flat:
        style(ax)
    rows = lgbm_panel(axes[0, 0], "A: LightGBM", df, folds, exps, chosen)
    rows += lgbm_panel(axes[0, 1], "C: LightGBM", df, folds, exps, chosen)
    ridge_panel(axes[1, 0], "A: augmented ridge", df, folds, exps, chosen)
    ridge_panel(axes[1, 1], "C: augmented ridge", df, folds, exps, chosen)
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=INK)
    axes[1, 0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Convergence on the primary split. Top: LightGBM loss per boosting round (thin = one outer fold). "
                 "Bottom: Huber-ridge optimiser.", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png", dpi=150)

    lg = pd.DataFrame(rows)
    print("LightGBM (Huber, unweighted): chosen round vs held-out optimum, per outer fold")
    print(lg.round(3).to_string(index=False))

    rc = ridge_grid_check(df, folds, exps)
    rc.to_csv(OUT / "convergence_ridge_grid.csv", index=False)
    print("\nHuber ridge: every configuration of every tuning grid, on every outer training set")
    print(rc.groupby("model").agg(fits=("converged", "size"), converged=("converged", "sum"),
                                  max_iter=("n_iter", "max"), max_rel_grad=("rel_grad", "max"),
                                  median_rel_grad=("rel_grad", "median")).to_string())
    if (~rc["converged"]).any():
        print(rc[~rc["converged"]].to_string(index=False))

    edges = pd.read_csv(OUT / "grid_edges.csv")
    print("\nGrid edges (grids with >= 3 values; flagged if >= 50% of picks are at an edge)")
    print(edges.assign(flag=np.where(edges["edge_share"] >= 0.5, "EDGE", "")).to_string(index=False))


if __name__ == "__main__":
    main()
