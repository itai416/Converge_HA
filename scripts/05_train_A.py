"""Model A (sequence only) and its baselines under repeated nested grouped CV.

Every configuration (model x loss x weighting) is run on N_REPEATS fold assignments (repeat 0 = folds.csv),
LightGBM also with N_SEEDS seeds. Outputs in results/model_A/ (see src/models/run.py).
Usage: python scripts/05_train_A.py [esm2 size, default 35M] [n_repeats, default 5] [n_seeds, default 3]
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # parallelism comes from running CV jobs side by side
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.splits import repeated_complex_folds  # noqa: E402
from src.models.regressors import AugmentedRidge, MeanBaseline, PCALightGBM  # noqa: E402
from src.models.run import run_experiments  # noqa: E402

OUT = ROOT / "results" / "model_A"
KEY = ["complex", "Mutation(s)_cleaned"]
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]


def load(size):
    dd = pd.read_parquet(ROOT / "data" / "processed" / "skempi_abag_dedup.parquet")
    v1 = dd[~dd["censored"] & (dd["n_mut"] == 1)][KEY + ["ddG", "iMutation_Location(s)", "wt_aa", "mut_aa"]]
    esm = pd.read_parquet(ROOT / "data" / "processed" / f"esm2_{size}.parquet")
    df = v1.merge(esm, on=KEY, validate="1:1").reset_index(drop=True)
    assert len(df) == len(v1)
    return dd, df


def experiments(df):
    emb = [c for c in df.columns if c.startswith(("wt_emb_", "diff_emb_"))]
    loc_sub = pd.get_dummies(df[["iMutation_Location(s)", "wt_aa", "mut_aa"]].astype(str), dtype=float)
    return {
        # name: (model class, fixed params, tuning grid, feature frame)
        "mean": (MeanBaseline, {}, {}, df[["llr"]]),
        "zero-shot LLR (calibrated)": (AugmentedRidge, {"boost_cols": ["llr"], "boost": 1.0, "alpha": 1e-3}, {}, df[["llr"]]),
        "location + substitution": (AugmentedRidge, {}, {"alpha": ALPHAS}, loc_sub),
        "A: augmented ridge": (AugmentedRidge, {"boost_cols": ["llr"]},
                               {"alpha": ALPHAS, "boost": [1.0, 10.0], "n_pca": [None, 16, 64]}, df[["llr"] + emb]),
        "A: LightGBM + PCA": (PCALightGBM, {"keep_cols": ["llr"]},
                              {"n_pca": [8, 16, 32], "num_leaves": [4, 7]}, df[["llr"] + emb]),
    }


def main(size="35M", n_repeats="5", n_seeds="3"):
    dd, df = load(size)
    rep_folds = repeated_complex_folds(dd, int(n_repeats))
    check = pd.read_csv(ROOT / "data" / "processed" / "folds.csv").merge(rep_folds, on="complex")
    assert (check["fold_complex"] == check["rep0"]).all(), "repeat 0 must equal folds.csv"
    summary = run_experiments(experiments(df), df, rep_folds, OUT, n_seeds=int(n_seeds))
    pd.set_option("display.width", 250)
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
