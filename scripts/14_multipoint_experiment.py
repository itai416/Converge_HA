"""Does adding multi-point rows to the training set improve model C?

Rows: 668 single-point (v1) + 272 uncensored multi-point rows. Per-site features (geometry, ESM-2 llr and embeddings) are pooled over the
sites of a mutation set; with `sum` pooling a linear model is exactly additive over sites (a single-point row is unchanged).
All variants are C (augmented ridge, Huber, same grid as 06). Folds are the 5 fold assignments of the earlier ablations; complexes that
only have multi-point rows are spread over the folds. Test scoring is separate for single-point rows (primary, paired with the
singles-only model on identical rows) and for multi-point rows.

  singles only   train on v1 only; multi-point rows are scored with the sum-pooled features = the additivity baseline
  + multi        train on v1 + multi, unweighted
  + multi, 1/sqrt(n)   same, with complex weights (one complex, 2B2X, has 86 of the 272 multi-point rows)
Usage: python scripts/14_multipoint_experiment.py [n_repeats]
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
from src.eval.metrics import bootstrap, metrics  # noqa: E402
from src.models.cv import nested_cv  # noqa: E402
from src.models.regressors import AugmentedRidge  # noqa: E402

abc = importlib.import_module("06_ablation_ABC")
KEY, G = abc.KEY, abc.GEOMETRY
OUT = ROOT / "results" / "multipoint"


def build(pool):
    dd, singles = abc.load("35M")
    emb = [c for c in singles.columns if c.startswith(("wt_emb_", "diff_emb_"))]
    feat = ["llr"] + G + emb
    singles = singles.assign(n_sites=1, is_multi=False)
    sites = pd.read_parquet(ROOT / "data" / "processed" / "multi_sites.parquet")
    agg = sites.groupby(KEY)[feat].agg("sum" if pool == "sum" else "mean").reset_index()
    agg["n_sites"] = sites.groupby(KEY).size().values
    multi = dd[~dd["censored"] & (dd["n_mut"] > 1)][KEY + ["ddG"]].merge(agg, on=KEY, validate="1:1").assign(is_multi=True)
    df = pd.concat([singles[KEY + ["ddG", "n_sites", "is_multi"] + feat], multi[KEY + ["ddG", "n_sites", "is_multi"] + feat]],
                   ignore_index=True)
    return dd, df, feat, emb


def folds_for(dd, df, n_repeats):
    rf = repeated_complex_folds(dd, n_repeats).set_index("complex")
    out = {}
    for r in rf.columns:
        f = rf[r].copy()
        rng = np.random.default_rng(int(r[3:]))
        for c in sorted(set(df["complex"]) - set(f.index)):  # complexes with only multi-point rows
            f[c] = int(rng.integers(5))
        out[r] = df["complex"].map(f).values
    return out


def job(name, X, df, fold, train_mask, w):
    make = lambda **p: AugmentedRidge(boost_cols=["llr"], scalar_cols=G, loss="huber", **p)  # noqa: E731
    pred, chosen = nested_cv(make, abc.RIDGE_GRID, X, df["ddG"].values, df["complex"], fold, w_scheme=w, train_mask=train_mask)
    return pred, chosen


def main(n_repeats="5"):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks, meta = [], []
    for pool in ["sum", "mean"]:
        dd, df, feat, _ = build(pool)
        folds = folds_for(dd, df, int(n_repeats))
        X = df[feat + (["n_sites"] if pool == "mean" else [])]
        variants = [("singles only", ~df["is_multi"].values, "none"), ("+ multi", np.ones(len(df), bool), "none"),
                    ("+ multi, 1/sqrt(n)", np.ones(len(df), bool), "sqrt")]
        for vname, mask, w in variants:
            for r, fold in folds.items():
                tasks.append((f"{vname} [{pool} pooling]", X, df, fold, mask, w))
                meta.append((f"{vname} [{pool} pooling]", r, df))
    print(f"{len(tasks)} runs", flush=True)
    res = Parallel(n_jobs=15, verbose=5)(delayed(job)(*t) for t in tasks)
    rows, oof = [], {}
    for (name, r, df), (pred, _) in zip(meta, res):
        d = df.assign(pred=pred)
        for part, sub in [("single-point test rows", d[~d["is_multi"]]), ("multi-point test rows", d[d["is_multi"]])]:
            rows.append({"model": name, "repeat": r, "test": part, **metrics(sub)})
        if r == "rep0":
            oof[name] = d[KEY + ["ddG", "is_multi"]].assign(pred=pred)
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "runs.csv", index=False)
    pd.set_option("display.width", 250)
    summ = runs.groupby(["test", "model"], sort=False)[["per_complex_spearman", "pooled_pearson", "rmse"]].agg(["mean", "std"]).round(3)
    summ.to_csv(OUT / "summary.csv")
    print(summ.to_string())
    ref = {t: runs[(runs.model == "singles only [sum pooling]") & (runs.test == t)].set_index("repeat")["per_complex_spearman"]
           for t in runs.test.unique()}
    pr = []
    for (t, m), g in runs.groupby(["test", "model"]):
        d = g.set_index("repeat")["per_complex_spearman"] - ref[t]
        pr.append({"test": t, "model": m, "vs singles-only (sum)": round(d.mean(), 3), "min": round(d.min(), 3),
                   "max": round(d.max(), 3), "repeats_better": f"{int((d > 0).sum())}/{len(d)}"})
    pr = pd.DataFrame(pr)
    pr.to_csv(OUT / "paired.csv", index=False)
    print("\npaired per-complex Spearman vs the singles-only model on identical rows/folds:\n" + pr.to_string(index=False))
    pd.concat(oof, names=["model"]).reset_index(0).to_parquet(OUT / "oof_rep0.parquet", index=False)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(*sys.argv[1:])
