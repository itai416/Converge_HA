"""Feature table, tuning grids and column groups of the single-point models (used by scripts/12_hypotheses.py).

Note: scripts/06_ablation_ABC.py still defines its own copy of these grids; the two must be kept in sync.
"""
import pandas as pd

from src.data.io import KEY, PROCESSED, is_v1, load_dedup

ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]          # embedding models
ALPHAS_LOW_DIM = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]  # location + substitution, geometry only
GEOMETRY = ["on_antibody", "rsa_complex", "rsa_unbound", "dsasa", "partner_atoms_4.5", "partner_atoms_8",
            "min_dist_partner", "cross_hbonds", "cross_saltbridges", "bfactor"]
RIDGE_GRID = {"alpha": ALPHAS, "boost": [1.0, 10.0], "n_pca": [None, 16, 64]}


def load(size="35M"):
    """(deduplicated table, v1 rows with their ESM-2 and geometry features)."""
    dd = load_dedup()
    v1 = dd[is_v1(dd)][KEY + ["ddG", "iMutation_Location(s)", "wt_aa", "mut_aa"]]
    df = (v1.merge(pd.read_parquet(PROCESSED / f"esm2_{size}.parquet"), on=KEY, validate="1:1")
          .merge(pd.read_parquet(PROCESSED / "geom_features.parquet"), on=KEY, validate="1:1"))
    assert len(df) == len(v1)
    return dd, df.reset_index(drop=True)


def emb_cols(df):
    """ESM-2 site embedding columns."""
    return [c for c in df.columns if c.startswith(("wt_emb_", "diff_emb_"))]
