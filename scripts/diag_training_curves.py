"""Training-curve diagnostics for model A on the primary split (repeat 0 = folds.csv).

LightGBM: loss per boosting round on the training rows and on the held-out outer fold, refit with the
          hyperparameters the inner CV chose in that fold (seed 0). The held-out curve is a diagnostic only;
          it was never used for model selection.
Ridge (Huber): objective per L-BFGS iteration for the chosen configurations, plus a convergence check over
          every configuration of the tuning grid. MSE ridge is solved in closed form, so it has no curve.
Outputs: results/model_A/training_curves_lgbm.png, training_curves_ridge.png
"""
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
from src.models.regressors import AugmentedRidge, PCALightGBM  # noqa: E402
import importlib  # noqa: E402

trainA = importlib.import_module("05_train_A")

OUT = ROOT / "results" / "model_A"
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
LOSS_LABEL = {"huber": "Huber loss (δ = 1)", "mse": "mean squared error"}


def style(ax):
    ax.grid(True, color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#bdbcb6")
    ax.tick_params(colors=MUTED, labelsize=8)


def lgbm_curves(df, folds, exps, chosen):
    cls, fixed, _, X = exps["A: LightGBM + PCA"]
    y = df["ddG"].values
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    report = []
    for i, loss in enumerate(["huber", "mse"]):
        for j, w in enumerate(["none", "sqrt"]):
            ax = axes[i, j]
            style(ax)
            trains, tests = [], []
            for k in range(5):
                tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
                p = chosen[f"A: LightGBM + PCA | {loss} | {w} | rep0 | 0"][str(k)]
                g = df["complex"].iloc[tr]
                sw = np.ones(len(tr)) if w == "none" else 1 / np.sqrt(g.map(g.value_counts()).values)
                m = cls(**fixed, loss=loss, seed=0, **p).fit(
                    X.iloc[tr], y[tr], sample_weight=sw, eval_sets=[(X.iloc[tr], y[tr]), (X.iloc[te], y[te])])
                key = "huber" if loss == "huber" else "l2"
                a, b = np.array(m.evals_result_["valid_0"][key]), np.array(m.evals_result_["valid_1"][key])
                trains.append(a), tests.append(b)
                ax.plot(a, color=BLUE, lw=0.8, alpha=0.35)
                ax.plot(b, color=ORANGE, lw=0.8, alpha=0.35)
                report.append({"loss": loss, "weighting": w, "fold": k, "best_round": int(b.argmin()) + 1,
                               "heldout_min": b.min(), "heldout_final": b[-1], "train_final": a[-1]})
            ax.plot(np.mean(trains, 0), color=BLUE, lw=2, label="training rows (mean of 5 folds)")
            ax.plot(np.mean(tests, 0), color=ORANGE, lw=2, label="held-out fold (mean of 5 folds)")
            ax.set_title(f"loss = {loss}, weighting = {w}", fontsize=10, color=INK, loc="left")
            ax.set_ylabel(LOSS_LABEL[loss], fontsize=9, color=MUTED)
            if i == 1:
                ax.set_xlabel("boosting round", fontsize=9, color=MUTED)
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Model A, LightGBM + PCA: loss per boosting round (thin lines = individual outer folds)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "training_curves_lgbm.png", dpi=150)
    return pd.DataFrame(report)


def _fit_ridge(cls, fixed, params, X, y, sw):
    m = cls(**fixed, loss="huber", track_history=True, **params).fit(X, y, sample_weight=sw)
    return m.history_, m.converged_, m.n_iter_


def ridge_curves(df, folds, exps, chosen):
    y = df["ddG"].values
    models = ["A: augmented ridge", "location + substitution"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True)
    for i, name in enumerate(models):
        cls, fixed, _, X = exps[name]
        for j, w in enumerate(["none", "sqrt"]):
            ax = axes[i, j]
            style(ax)
            for k in range(5):
                tr = np.where(folds != k)[0]
                g = df["complex"].iloc[tr]
                sw = np.ones(len(tr)) if w == "none" else 1 / np.sqrt(g.map(g.value_counts()).values)
                p = chosen[f"{name} | huber | {w} | rep0 | None"][str(k)]
                h, ok, n = _fit_ridge(cls, fixed, p, X.iloc[tr], y[tr], sw)
                h = np.array(h)
                gap = (h - h[-1]) / abs(h[-1]) + 1e-12  # relative distance to the final objective
                ax.semilogy(np.arange(1, len(h) + 1), gap, color=BLUE, lw=1.2, alpha=0.8,
                            label="one outer fold" if k == 0 else None)
            ax.set_title(f"{name}, weighting = {w}", fontsize=10, color=INK, loc="left")
            ax.set_xlabel("L-BFGS iteration", fontsize=9, color=MUTED)
            if j == 0:
                ax.set_ylabel("relative gap to final objective", fontsize=9, color=MUTED)
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Huber ridge: optimiser convergence for the chosen configuration in each outer fold",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "training_curves_ridge.png", dpi=150)

    # convergence of every configuration in the tuning grid, on every outer training set
    jobs = []
    for name in models:
        cls, fixed, grid, X = exps[name]
        configs = [dict(zip(grid, v)) for v in __import__("itertools").product(*grid.values())]
        for w in ["none", "sqrt"]:
            for k in range(5):
                tr = np.where(folds != k)[0]
                g = df["complex"].iloc[tr]
                sw = np.ones(len(tr)) if w == "none" else 1 / np.sqrt(g.map(g.value_counts()).values)
                for p in configs:
                    jobs.append((name, w, k, p, (cls, fixed, p, X.iloc[tr], y[tr], sw)))
    res = Parallel(n_jobs=15)(delayed(_fit_ridge)(*j[-1]) for j in jobs)
    return pd.DataFrame([{"model": j[0], "weighting": j[1], "fold": j[2], "params": str(j[3]),
                          "converged": ok, "n_iter": n} for j, (_, ok, n) in zip(jobs, res)])


def main():
    _, df = trainA.load("35M")
    folds = df[["complex"]].merge(pd.read_csv(ROOT / "data" / "processed" / "folds.csv"), on="complex")[
        "fold_complex"].values
    exps = trainA.experiments(df)
    chosen = json.loads((OUT / "chosen_params.json").read_text())

    lg = lgbm_curves(df, folds, exps, chosen)
    print("LightGBM: held-out loss, best round vs final (300) round")
    print(lg.groupby(["loss", "weighting"]).agg(best_round_median=("best_round", "median"),
                                                 heldout_min=("heldout_min", "mean"),
                                                 heldout_final=("heldout_final", "mean"),
                                                 train_final=("train_final", "mean")).round(3).to_string())
    rc = ridge_curves(df, folds, exps, chosen)
    print("\nHuber ridge: convergence over the full tuning grid")
    print(rc.groupby(["model", "weighting"]).agg(fits=("converged", "size"), converged=("converged", "sum"),
                                                  max_iter=("n_iter", "max")).to_string())
    if (~rc["converged"]).any():
        print(rc[~rc["converged"]].to_string())


if __name__ == "__main__":
    main()
