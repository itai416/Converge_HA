"""Ablation ladder, structure rungs: does a learned structure model (ESM-IF1) add anything beyond hand-crafted geometry?

C              A + geometry (reference, same as 06_ablation_ABC.py)
D              C + ESM-IF1 scores (site / whole-chain log-likelihood ratios in complex, unbound and their difference)
E              D + ESM-IF1 site embeddings (complex and unbound, 512 dims each)
structure      geometry + ESM-IF1 scores, no sequence model (the "structure only" arm, now with a learned component)
geometry only  reference
zero-shot      the ESM-IF1 interface score alone, calibrated to kcal/mol with a 2-parameter line (no training of a model)

Same protocol as 06: Huber loss, unweighted, 5 fold assignments (repeat 0 = folds.csv), LightGBM x 3 seeds, all
hyperparameters (including boosting rounds and PCA size) tuned on inner folds; grid edges reported in grid_edges.csv.
Needs data/processed/esmif1.parquet (scripts/08_esmif1.py or notebooks/02_colab_embeddings.ipynb).
Usage: python scripts/09_ablation_DE.py [esm2 size, default 35M; 650M once esm2_650M.parquet exists] [n_repeats] [n_seeds]
"""
import importlib
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.data.splits import repeated_complex_folds  # noqa: E402
from src.models.regressors import AugmentedRidge, PCALightGBM  # noqa: E402
from src.models.run import run_experiments  # noqa: E402

abc = importlib.import_module("06_ablation_ABC")
KEY = abc.KEY
IF_SCORES = [f"if_{k}_{c}" for k in ("site_llr", "seq_llr") for c in ("complex", "unbound", "interface")]


def load(size):
    dd, df = abc.load(size)
    esmif = pd.read_parquet(ROOT / "data" / "processed" / "esmif1.parquet")
    df = df.merge(esmif, on=KEY, validate="1:1").reset_index(drop=True)
    assert len(df) == len(dd[~dd["censored"] & (dd["n_mut"] == 1)]), "esmif1.parquet does not cover every v1 row"
    return dd, df


def experiments(df):
    emb = [c for c in df.columns if c.startswith(("wt_emb_", "diff_emb_"))]
    if_emb = [c for c in df.columns if c.startswith("if_emb_")]
    G, R, L = abc.GEOMETRY, abc.RIDGE_GRID, abc.LGBM_GRID
    exps = {
        "zero-shot ESM-IF1 interface (calibrated)": (
            AugmentedRidge, {"boost_cols": ["if_seq_llr_interface"], "boost": 1.0, "alpha": 1e-3}, {},
            df[["if_seq_llr_interface"]]),
        "geometry only: ridge": (AugmentedRidge, {"scalar_cols": G}, {"alpha": abc.ALPHAS_LOW_DIM}, df[G]),
        "structure (geometry + ESM-IF1 scores): ridge": (
            AugmentedRidge, {"scalar_cols": G + IF_SCORES}, {"alpha": abc.ALPHAS_LOW_DIM}, df[G + IF_SCORES]),
        "structure (geometry + ESM-IF1 scores): LightGBM": (
            PCALightGBM, {"keep_cols": G + IF_SCORES}, {"num_leaves": [4, 7], "n_estimators": abc.ROUNDS},
            df[G + IF_SCORES]),
    }
    # embedding columns (ESM-2, and for E also ESM-IF1) form the block that ridge may use in full or compress with PCA
    for tag, scalars, embs in [("C", G, emb), ("D", G + IF_SCORES, emb), ("E", G + IF_SCORES, emb + if_emb)]:
        X = df[["llr"] + scalars + embs]
        exps[f"{tag}: augmented ridge"] = (AugmentedRidge, {"boost_cols": ["llr"], "scalar_cols": scalars}, R, X)
        exps[f"{tag}: LightGBM"] = (PCALightGBM, {"keep_cols": ["llr"] + scalars}, L, X)
    return exps


def paired(runs):
    r = runs.groupby(["model", "repeat"])["per_complex_spearman"].mean().unstack("model")
    pairs = [("D: augmented ridge", "C: augmented ridge"), ("D: LightGBM", "C: LightGBM"),
             ("E: augmented ridge", "D: augmented ridge"), ("E: LightGBM", "D: LightGBM"),
             ("structure (geometry + ESM-IF1 scores): ridge", "geometry only: ridge"),
             ("D: augmented ridge", "structure (geometry + ESM-IF1 scores): ridge")]
    rows = []
    for a, b in pairs:
        d = r[a] - r[b]
        rows.append({"comparison": f"{a}  -  {b}", "mean_diff": d.mean(), "min": d.min(), "max": d.max(),
                     "repeats_better": f"{int((d > 0).sum())}/{len(d)}"})
    return pd.DataFrame(rows).round(3)


def main(size="35M", n_repeats="5", n_seeds="3"):
    out = ROOT / "results" / f"ablation_DE_esm2_{size}"
    dd, df = load(size)
    rep_folds = repeated_complex_folds(dd, int(n_repeats))
    summary = run_experiments(experiments(df), df, rep_folds, out, n_seeds=int(n_seeds),
                              losses=["huber"], weightings=["none"])
    pd.set_option("display.width", 250)
    print(summary.round(3).to_string())
    p = paired(pd.read_csv(out / "runs.csv"))
    p.to_csv(out / "paired.csv", index=False)
    print("\npaired per-complex Spearman differences (same folds):\n" + p.to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
