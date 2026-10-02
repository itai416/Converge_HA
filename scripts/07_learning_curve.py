"""Learning curve: does performance still improve with more training complexes?

In every outer fold of every fold assignment (repeats from repeated_complex_folds), the model is trained on a random
fraction of the training *complexes* (whole complexes, so no within-complex leakage) and scored on the untouched test fold.
Fractions < 1 are drawn 3 times. Out-of-fold predictions of one (repeat, fraction, draw) cover every row once, so the
metrics are computed exactly as in the main results.

Hyperparameters are fixed to each model's most frequently chosen configuration in the ablation (chosen_params.json);
re-tuning at every size would need ~40,000 extra fits. A smaller training set may prefer a stronger penalty, so the
small-fraction points are, if anything, slightly pessimistic.
Outputs in results/learning_curve/: runs.csv, learning_curve.png
"""
import importlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.data.splits import repeated_complex_folds  # noqa: E402
from src.eval.metrics import MIN_ROWS, metrics  # noqa: E402

ablation = importlib.import_module("06_ablation_ABC")
OUT = ROOT / "results" / "learning_curve"
MODELS = ["C: augmented ridge", "geometry only: ridge", "A: LightGBM"]
FRACTIONS = [0.25, 0.5, 0.75, 1.0]
N_DRAWS = 3
COLORS = {"C: augmented ridge": "#2a78d6", "geometry only: ridge": "#1baf7a", "A: LightGBM": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def most_chosen(chosen, name):
    picks = [tuple(sorted(p.items())) for key, folds in chosen.items() if key.split(" | ")[0] == name
             for p in folds.values()]
    return dict(Counter(picks).most_common(1)[0][0])


def _fit(cls, fixed, params, X, y, groups, tr, te, frac, seed):
    rng = np.random.default_rng(seed)
    cx = np.unique(groups[tr])
    keep = rng.choice(cx, size=max(2, int(round(frac * len(cx)))), replace=False) if frac < 1 else cx
    sub = tr[np.isin(groups[tr], keep)]
    m = cls(**fixed, loss="huber", **params).fit(X.iloc[sub], y[sub])
    # in-sample per-complex Spearman on the rows actually trained on (for the train/test gap)
    ptr = m.predict(X.iloc[sub])
    r = [spearmanr(ptr[groups[sub] == c], y[sub][groups[sub] == c])[0] for c in keep if (groups[sub] == c).sum() >= MIN_ROWS]
    return m.predict(X.iloc[te]), len(keep), len(sub), float(np.nanmean(r)) if r else np.nan


def main(n_repeats="5"):
    OUT.mkdir(parents=True, exist_ok=True)
    dd, df = ablation.load("35M")
    exps = ablation.experiments(df)
    chosen = json.loads((ROOT / "results" / "ablation_ABC" / "chosen_params.json").read_text())
    rep_folds = repeated_complex_folds(dd, int(n_repeats))
    folds = df[["complex"]].merge(rep_folds, on="complex", how="left")
    y, groups = df["ddG"].values, df["complex"].values
    reps = [c for c in rep_folds.columns if c.startswith("rep")]

    tasks = []
    for name in MODELS:
        cls, fixed, _, X = exps[name]
        params = most_chosen(chosen, name)
        if "seed" in cls.__init__.__code__.co_varnames:
            params["seed"] = 0
        print(f"{name}: fixed params {params}")
        for r in reps:
            for frac in FRACTIONS:
                for d in range(N_DRAWS if frac < 1 else 1):
                    for k in range(5):
                        tr, te = np.where(folds[r] != k)[0], np.where(folds[r] == k)[0]
                        seed = reps.index(r) * 10000 + int(frac * 100) * 100 + d * 10 + k  # same draws for every model
                        tasks.append(((name, r, frac, d, k, te), (cls, fixed, params, X, y, groups, tr, te, frac, seed)))
    print(f"{len(tasks)} fits")
    res = Parallel(n_jobs=15, verbose=2)(delayed(_fit)(*a) for _, a in tasks)

    preds, sizes = {}, {}
    for (key, _), (p, n_cx, n_rows, train_rho) in zip(tasks, res):
        name, r, frac, d, k, te = key
        run = (name, r, frac, d)
        preds.setdefault(run, np.full(len(y), np.nan))[te] = p
        sizes.setdefault(run, []).append((n_cx, n_rows, train_rho))
    rows = []
    base = df[["complex", "ddG"]]
    for run, p in preds.items():
        s = np.array(sizes[run])
        rows.append({"model": run[0], "repeat": run[1], "fraction": run[2], "draw": run[3],
                     "train_complexes": s[:, 0].mean(), "train_rows": s[:, 1].mean(),
                     "train_per_complex_spearman": np.nanmean(s[:, 2]), **metrics(base.assign(pred=p))})
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "runs.csv", index=False)

    summ = runs.groupby(["model", "fraction"]).agg(
        train_complexes=("train_complexes", "mean"), train_rows=("train_rows", "mean"),
        test_rho=("per_complex_spearman", "mean"), test_rho_sd=("per_complex_spearman", "std"),
        train_rho=("train_per_complex_spearman", "mean"), pooled=("pooled_pearson", "mean"),
        rmse=("rmse", "mean"), rmse_sd=("rmse", "std"))
    print(summ.round(3).to_string())
    plot(summ)


def plot(summ):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.4), gridspec_kw={"width_ratios": [1, 1, 1]})
    for ax in axes:
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color("#bdbcb6")
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel("training complexes per fold (mean)", fontsize=9, color=MUTED)
    for name in MODELS:
        s = summ.loc[name]
        x, c = s["train_complexes"].values, COLORS[name]
        axes[0].plot(x, s["test_rho"], "-o", color=c, lw=2, ms=6, mec="white", mew=1.5, label=name)
        axes[0].fill_between(x, s["test_rho"] - s["test_rho_sd"], s["test_rho"] + s["test_rho_sd"], color=c, alpha=0.12, lw=0)
        axes[1].plot(x, s["rmse"], "-o", color=c, lw=2, ms=6, mec="white", mew=1.5, label=name)
        axes[1].fill_between(x, s["rmse"] - s["rmse_sd"], s["rmse"] + s["rmse_sd"], color=c, alpha=0.12, lw=0)
        axes[2].plot(x, s["train_rho"], "--", color=c, lw=1.5, label=f"{name}, training rows")
        axes[2].plot(x, s["test_rho"], "-o", color=c, lw=2, ms=6, mec="white", mew=1.5, label=f"{name}, held-out")
    axes[0].set_title("per-complex Spearman, held-out (higher is better)", fontsize=10, color=INK, loc="left")
    axes[1].set_title("RMSE, held-out, kcal/mol (lower is better)", fontsize=10, color=INK, loc="left")
    axes[2].set_title("training vs held-out per-complex Spearman", fontsize=10, color=INK, loc="left")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    axes[2].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Learning curve: random subsets of training complexes (bands = ±1 std over 5 fold assignments × draws)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "learning_curve.png", dpi=150)


if __name__ == "__main__":
    main(*sys.argv[1:])
