"""Paths, key columns and row filters shared by the pipeline."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
KEY = ["complex", "Mutation(s)_cleaned"]  # identifies one row of the deduplicated table


def is_v1(df: pd.DataFrame) -> pd.Series:
    """Mask of the v1 set: exact (uncensored) single-point rows."""
    return ~df["censored"] & (df["n_mut"] == 1)


def load_dedup() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "skempi_abag_dedup.parquet")


def load_oof_frame(ann_cols: list, geom_cols: list = None, fold_cols: list = ("antigen_seen",)):
    """(oof, base) for the error analysis, on the primary split (repeat 0).

    oof: the out-of-fold predictions of the D/E ablation, one column per model. base: the same rows in the same order
    with ddG, the columns `ann_cols` of the deduplicated table, the geometry features (all, or `geom_cols`), the
    per-complex `fold_cols` of folds.csv and `side` (antibody / antigen).
    """
    oof = pd.read_parquet(RESULTS / "ablation_DE_esm2_35M" / "oof_rep0.parquet")
    dd = load_dedup()
    geom = pd.read_parquet(PROCESSED / "geom_features.parquet")
    folds = pd.read_csv(PROCESSED / "folds.csv")
    base = (oof[KEY + ["ddG"]].merge(dd[is_v1(dd)][KEY + ann_cols], on=KEY, validate="1:1")
            .merge(geom if geom_cols is None else geom[KEY + geom_cols], on=KEY, validate="1:1")
            .merge(folds[["complex", *fold_cols]], on="complex", validate="m:1"))
    base["side"] = np.where(base["on_antibody"] == 1, "antibody", "antigen")
    return oof, base
