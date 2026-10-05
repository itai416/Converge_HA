"""Do right-censored rows help model C when trained with a Tobit-style (one-sided Huber) loss?

Training set: the 668 exact single-point rows (v1) + 24 usable censored single-point rows (true ddG >= bound; see 15_censored_features.py).
Test scoring is always on the exact v1 rows only, with the same folds as the earlier ablations, so every comparison is paired with C.
All variants are C (augmented ridge, Huber, grid of 06); the inner CV also scores on exact rows only, so censored rows never influence which
configuration is chosen.

  C (exact rows only)           reference
  + censored, one-sided Huber   Tobit-style: a censored row costs huber(max(0, bound - prediction))
  + censored, bound as label    naive control: the bound is treated as an exact ddG
  (each of the two also with 1/sqrt(n) complex weights, since one complex, 4I77, has 14 of the 24 censored rows)
Extra diagnostics on the exact test rows with ddG > 2 (where the model under-predicts) and on the censored test rows (how often is the
prediction below the bound?).
Usage: python scripts/16_censored_experiment.py [n_repeats]
"""
import importlib
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.data.splits import repeated_complex_folds  # noqa: E402
from src.eval.metrics import metrics  # noqa: E402
from src.models.cv import nested_cv  # noqa: E402
from src.models.regressors import AugmentedRidge  # noqa: E402

abc = importlib.import_module("06_ablation_ABC")
KEY, G = abc.KEY, abc.GEOMETRY
OUT = ROOT / "results" / "censored"


def build():
    dd, singles = abc.load("35M")
    cens = pd.read_parquet(ROOT / "data" / "processed" / "censored_features.parquet")
    emb = [c for c in singles.columns if c.startswith(("wt_emb_", "diff_emb_"))]
    feat = ["llr"] + G + emb
    a = singles[KEY + ["ddG"] + feat].assign(is_cens=0)
    b = cens.rename(columns={"ddG_bound": "ddG"})[KEY + ["ddG"] + feat].assign(is_cens=1)
    return dd, pd.concat([a, b], ignore_index=True), feat


def folds_for(dd, df, n_repeats):
    rf = repeated_complex_folds(dd, n_repeats).set_index("complex")
    out = {}
    for r in rf.columns:
        f = rf[r].copy()
        rng = np.random.default_rng(int(r[3:]))
        for c in sorted(set(df["complex"]) - set(f.index)):  # complexes with only censored rows
            f[c] = int(rng.integers(5))
        out[r] = df["complex"].map(f).values
    return out


def job(X, df, fold, train_mask, w, cens_col):
    make = lambda **p: AugmentedRidge(boost_cols=["llr"], scalar_cols=G, loss="huber", cens_col=cens_col, **p)  # noqa: E731
    return nested_cv(make, abc.RIDGE_GRID, X, df["ddG"].values, df["complex"], fold, w_scheme=w, train_mask=train_mask,
                     score_mask=(df["is_cens"] == 0).values)


def main(n_repeats="5"):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    dd, df, feat = build()
    folds = folds_for(dd, df, int(n_repeats))
    exact = (df["is_cens"] == 0).values
    allrows = np.ones(len(df), bool)
    Xc = df[feat + ["is_cens"]]  # the model drops is_cens (cens_col); without cens_col it would be a feature, so the naive control uses X
    X = df[feat]
    variants = {"C (exact rows only)": (X, exact, "none", None),
                "+ censored, one-sided Huber (Tobit)": (Xc, allrows, "none", "is_cens"),
                "+ censored, bound used as exact label": (X, allrows, "none", None),
                "+ censored, one-sided Huber, 1/sqrt(n)": (Xc, allrows, "sqrt", "is_cens"),
                "+ censored, bound as label, 1/sqrt(n)": (X, allrows, "sqrt", None)}
    tasks = [(name, r, v, fold) for name, v in variants.items() for r, fold in folds.items()]
    print(f"{len(tasks)} runs", flush=True)
    res = Parallel(n_jobs=15, verbose=5)(delayed(job)(v[0], df, fold, v[1], v[2], v[3]) for _, r, v, fold in tasks)
    rows = []
    for (name, r, _, _), (pred, _) in zip(tasks, res):
        d = df.assign(pred=pred)
        ex, ce = d[d["is_cens"] == 0], d[d["is_cens"] == 1]
        tail = ex[ex["ddG"] > 2]
        rows.append({"model": name, "repeat": r, **metrics(ex), "tail_mean_pred": tail["pred"].mean(),
                     "tail_rmse": float(np.sqrt(((tail["pred"] - tail["ddG"]) ** 2).mean())),
                     "cens_below_bound": float((ce["pred"] < ce["ddG"]).mean()),
                     "cens_mean_shortfall": float(np.maximum(ce["ddG"] - ce["pred"], 0).mean())})
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "runs.csv", index=False)
    cols = ["per_complex_spearman", "pooled_pearson", "rmse", "tail_mean_pred", "tail_rmse", "cens_below_bound", "cens_mean_shortfall"]
    summ = runs.groupby("model", sort=False)[cols].agg(["mean", "std"]).round(3)
    summ.to_csv(OUT / "summary.csv")
    pd.set_option("display.width", 250)
    print(summ.to_string())
    ref = runs[runs.model == "C (exact rows only)"].set_index("repeat")
    pr = []
    for m, g in runs.groupby("model", sort=False):
        g = g.set_index("repeat")
        d = g["per_complex_spearman"] - ref["per_complex_spearman"]
        pr.append({"model": m, "per-complex Spearman vs C": round(d.mean(), 3), "min": round(d.min(), 3), "max": round(d.max(), 3),
                   "repeats_better": f"{int((d > 0).sum())}/{len(d)}",
                   "tail RMSE vs C": round((g["tail_rmse"] - ref["tail_rmse"]).mean(), 3)})
    pr = pd.DataFrame(pr)
    pr.to_csv(OUT / "paired.csv", index=False)
    print("\npaired vs C on identical exact test rows/folds:\n" + pr.to_string(index=False))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(*sys.argv[1:])
