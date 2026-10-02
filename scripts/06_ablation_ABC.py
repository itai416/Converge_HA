"""Ablation ladder A -> B -> C (+ structure-only reference) under repeated nested grouped CV.

A          sequence only: ESM-2 LLR + site embeddings
B          A + distance of the mutated residue to the partner (min_dist_partner)
C          A + all structural geometry features of the wild-type complex
geometry   structural geometry only (no sequence model): the "structure only" arm

Every model is run as augmented ridge and as LightGBM, Huber loss, unweighted (the defaults agreed after
model A), on 5 fold assignments (repeat 0 = folds.csv) and 3 LightGBM seeds. The number of boosting rounds
is tuned on the inner folds. Outputs in results/ablation_ABC/ (see src/models/run.py) plus paired.csv.
Usage: python scripts/06_ablation_ABC.py [esm2 size, default 35M] [n_repeats, default 5] [n_seeds, default 3]
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.splits import repeated_complex_folds  # noqa: E402
from src.models.regressors import AugmentedRidge, PCALightGBM  # noqa: E402
from src.models.run import run_experiments  # noqa: E402

OUT = ROOT / "results" / "ablation_ABC"
KEY = ["complex", "Mutation(s)_cleaned"]
# Grids widened after the first run: geometry-only ridge picked the lowest alpha (1) in 19/25 folds and geometry-only
# LightGBM the fewest rounds (100) in 61/75, so the optimum lay outside the grid. grid_edges.csv checks this every run.
# Low alphas are only offered to the low-dimensional models: on ~960 embedding columns they make the Huber problem
# ill-conditioned (alpha=0.01 needs ~1,900 L-BFGS iterations) and those models never chose alpha near 1 anyway.
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]          # embedding models
ALPHAS_LOW_DIM = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]  # location + substitution, geometry only
ROUNDS = [25, 50, 100, 200, 300, 500]  # 500 and PCA 64 added: A LightGBM sat on the upper edges (32 in 81%, 300 in 61%)
DISTANCE = ["min_dist_partner"]
GEOMETRY = ["on_antibody", "rsa_complex", "rsa_unbound", "dsasa", "partner_atoms_4.5", "partner_atoms_8",
            "min_dist_partner", "cross_hbonds", "cross_saltbridges", "bfactor"]
LGBM_GRID = {"n_pca": [8, 16, 32, 64], "num_leaves": [4, 7], "n_estimators": ROUNDS}
RIDGE_GRID = {"alpha": ALPHAS, "boost": [1.0, 10.0], "n_pca": [None, 16, 64]}


def load(size):
    dd = pd.read_parquet(ROOT / "data" / "processed" / "skempi_abag_dedup.parquet")
    v1 = dd[~dd["censored"] & (dd["n_mut"] == 1)][KEY + ["ddG", "iMutation_Location(s)", "wt_aa", "mut_aa"]]
    esm = pd.read_parquet(ROOT / "data" / "processed" / f"esm2_{size}.parquet")
    geo = pd.read_parquet(ROOT / "data" / "processed" / "geom_features.parquet")
    df = v1.merge(esm, on=KEY, validate="1:1").merge(geo, on=KEY, validate="1:1").reset_index(drop=True)
    assert len(df) == len(v1)
    return dd, df


def experiments(df):
    emb = [c for c in df.columns if c.startswith(("wt_emb_", "diff_emb_"))]
    loc_sub = pd.get_dummies(df[["iMutation_Location(s)", "wt_aa", "mut_aa"]].astype(str), dtype=float)
    exps = {"location + substitution": (AugmentedRidge, {}, {"alpha": ALPHAS_LOW_DIM}, loc_sub)}
    for tag, extra in [("A", []), ("B", DISTANCE), ("C", GEOMETRY)]:
        X = df[["llr"] + extra + emb]
        exps[f"{tag}: augmented ridge"] = (AugmentedRidge, {"boost_cols": ["llr"], "scalar_cols": extra}, RIDGE_GRID, X)
        exps[f"{tag}: LightGBM"] = (PCALightGBM, {"keep_cols": ["llr"] + extra}, LGBM_GRID, X)
    exps["geometry only: ridge"] = (AugmentedRidge, {"scalar_cols": GEOMETRY}, {"alpha": ALPHAS_LOW_DIM}, df[GEOMETRY])
    exps["geometry only: LightGBM"] = (PCALightGBM, {"keep_cols": GEOMETRY},
                                       {"num_leaves": [4, 7], "n_estimators": ROUNDS}, df[GEOMETRY])
    return exps


def paired(runs):
    """Per repeat (seeds averaged), differences in per-complex Spearman between rungs of the ladder."""
    r = runs.groupby(["model", "repeat"])["per_complex_spearman"].mean().unstack("model")
    rows = []
    for reg in ["augmented ridge", "LightGBM"]:
        for a, b in [("B", "A"), ("C", "A"), ("C", "B")]:
            d = r[f"{a}: {reg}"] - r[f"{b}: {reg}"]
            rows.append({"comparison": f"{a} - {b} ({reg})", "mean_diff": d.mean(), "min": d.min(), "max": d.max(),
                         "repeats_better": f"{int((d > 0).sum())}/{len(d)}"})
    d = r["C: LightGBM"] - r["location + substitution"]
    rows.append({"comparison": "C LightGBM - location + substitution", "mean_diff": d.mean(), "min": d.min(),
                 "max": d.max(), "repeats_better": f"{int((d > 0).sum())}/{len(d)}"})
    return pd.DataFrame(rows).round(3)


def main(size="35M", n_repeats="5", n_seeds="3"):
    dd, df = load(size)
    rep_folds = repeated_complex_folds(dd, int(n_repeats))
    summary = run_experiments(experiments(df), df, rep_folds, OUT, n_seeds=int(n_seeds),
                              losses=["huber"], weightings=["none"])
    pd.set_option("display.width", 250)
    print(summary.round(3).to_string())
    p = paired(pd.read_csv(OUT / "runs.csv"))
    p.to_csv(OUT / "paired.csv", index=False)
    print("\npaired per-complex Spearman differences (same folds):\n" + p.to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
