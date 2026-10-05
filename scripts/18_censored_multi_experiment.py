"""Model C trained on all usable data: exact single-point + exact multi-point + right-censored (single and multi) rows.

Rows: 668 exact single-point, 272 exact multi-point, 24 censored single-point and 23 censored multi-point rows with a numeric lower
bound (ddG >= bound; see 15_ and 17_). Per-site features are pooled over the sites of a mutation set (sum or mean) as in
14_multipoint_experiment.py. Censored rows use the one-sided Huber (Tobit-style) loss, in training only; the inner CV scores exact
rows only; test scoring is on exact rows (single-point, multi-point, and both together), same 5 fold assignments as before.

  singles only                 reference (= model C)
  + multi                      exact multi-point rows added (as in section 6.6)
  + multi + censored           censored rows added with the one-sided loss
  + multi + censored, 1/sqrt(n) the same with complex weights
Usage: python scripts/18_censored_multi_experiment.py [n_repeats]
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
from src.eval.metrics import metrics  # noqa: E402
from src.models.cv import nested_cv  # noqa: E402
from src.models.regressors import AugmentedRidge  # noqa: E402

mp = importlib.import_module("14_multipoint_experiment")
abc = mp.abc
KEY, G = abc.KEY, abc.GEOMETRY
OUT = ROOT / "results" / "censored_multi"


def build(pool):
    dd, df, feat, _ = mp.build(pool)
    df = df.assign(is_cens=0)
    agg = "sum" if pool == "sum" else "mean"
    cs = pd.read_parquet(ROOT / "data" / "processed" / "censored_features.parquet")
    cs = cs.rename(columns={"ddG_bound": "ddG"})[KEY + ["ddG"] + feat].assign(n_sites=1, is_multi=False, is_cens=1)
    sites = pd.read_parquet(ROOT / "data" / "processed" / "censored_multi_sites.parquet")
    cm = sites.groupby(KEY).agg({**{c: agg for c in feat}, "ddG_bound": "first", "site": "size"}).reset_index()
    cm = cm.rename(columns={"ddG_bound": "ddG", "site": "n_sites"}).assign(is_multi=True, is_cens=1)[KEY + ["ddG"] + feat + ["n_sites", "is_multi", "is_cens"]]
    return dd, pd.concat([df, cs, cm], ignore_index=True), feat


def job(X, df, fold, train_mask, w):
    make = lambda **p: AugmentedRidge(boost_cols=["llr"], scalar_cols=G, loss="huber", cens_col="is_cens", **p)  # noqa: E731
    return nested_cv(make, abc.RIDGE_GRID, X, df["ddG"].values, df["complex"], fold, w_scheme=w, train_mask=train_mask,
                     score_mask=(df["is_cens"] == 0).values)


def main(n_repeats="5"):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = []
    for pool in ["mean", "sum"]:
        dd, df, feat = build(pool)
        folds = mp.folds_for(dd, df, int(n_repeats))
        X = df[feat + (["n_sites"] if pool == "mean" else []) + ["is_cens"]]
        exact, single = df["is_cens"].values == 0, ~df["is_multi"].values
        variants = [("singles only", exact & single, "none"), ("+ multi", exact, "none"),
                    ("+ multi + censored", np.ones(len(df), bool), "none"),
                    ("+ multi + censored, 1/sqrt(n)", np.ones(len(df), bool), "sqrt")]
        for vname, mask, w in variants:
            for r, fold in folds.items():
                tasks.append((f"{vname} [{pool} pooling]", r, df, X, fold, mask, w))
    print(f"{len(tasks)} runs", flush=True)
    res = Parallel(n_jobs=15, verbose=5)(delayed(job)(t[3], t[2], t[4], t[5], t[6]) for t in tasks)
    rows = []
    for (name, r, df, *_), (pred, _) in zip(tasks, res):
        d = df.assign(pred=pred)
        ex, ce = d[d["is_cens"] == 0], d[d["is_cens"] == 1]
        for part, sub in [("single-point test rows", ex[~ex["is_multi"]]), ("multi-point test rows", ex[ex["is_multi"]]),
                          ("all test rows (single + multi)", ex)]:
            rows.append({"model": name, "repeat": r, "test": part, **metrics(sub),
                         "cens_below_bound": float((ce["pred"] < ce["ddG"]).mean())})
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "runs.csv", index=False)
    pd.set_option("display.width", 250)
    summ = runs.groupby(["test", "model"], sort=False)[["per_complex_spearman", "pooled_pearson", "rmse"]].agg(["mean", "std"]).round(3)
    summ.to_csv(OUT / "summary.csv")
    print(summ.to_string())
    pr = []
    for pool in ["mean", "sum"]:
        ref = runs[runs.model == f"singles only [{pool} pooling]"]
        for (t, m), g in runs[runs.model.str.contains(pool)].groupby(["test", "model"], sort=False):
            d = g.set_index("repeat")["per_complex_spearman"] - ref[ref.test == t].set_index("repeat")["per_complex_spearman"]
            pr.append({"test": t, "model": m, "vs singles-only (same pooling)": round(d.mean(), 3), "min": round(d.min(), 3),
                       "max": round(d.max(), 3), "repeats_better": f"{int((d > 0).sum())}/{len(d)}"})
    pr = pd.DataFrame(pr)
    pr.to_csv(OUT / "paired.csv", index=False)
    print("\npaired per-complex Spearman vs the singles-only model (same pooling), identical rows/folds:\n" + pr.to_string(index=False))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(*sys.argv[1:])
