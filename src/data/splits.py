"""Cross-validation folds. Whole complexes always stay in one fold.

fold_complex  primary: StratifiedGroupKFold, group = complex, stratum = antigen_group.
              Related complexes (same antigen) are spread over folds, so a test complex can have
              a "seen" antigen (lead optimisation on a known target) or an "unseen" one.
fold_antigen  secondary, harder: GroupKFold, group = antigen_group. A test antigen is never in train.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

N_FOLDS = 5
warnings.filterwarnings("ignore", message="The least populated class")


def _is_v1(df):
    return ~df["censored"] & (df["n_mut"] == 1)


def _complex_folds(v1, seed):
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold = pd.Series(-1, index=v1.index)
    for k, (_, test) in enumerate(sgkf.split(v1, v1["antigen_group"], v1["complex"])):
        fold.iloc[test] = k
    return v1.assign(fold=fold).groupby("complex")["fold"].first()


def make_folds(dedup_df: pd.DataFrame, n_seeds: int = 200) -> tuple[pd.DataFrame, int]:
    """Return one row per complex with its folds, plus the chosen seed.

    Folds are balanced on v1 rows; the seed with the most even fold sizes (out of n_seeds) is kept.
    Complexes with no v1 rows get the fold of another complex of the same antigen, else the
    currently smallest fold, so every complex has a fold for the later extensions.
    """
    v1 = dedup_df[_is_v1(dedup_df)].reset_index(drop=True)
    cx = v1.groupby("complex").size()

    def imbalance(seed):
        f = _complex_folds(v1, seed)
        return np.bincount(f.values, weights=cx[f.index].values, minlength=N_FOLDS).std()

    seed = min(range(n_seeds), key=imbalance)
    by_complex = _complex_folds(v1, seed)

    info = dedup_df.groupby("complex").agg(antigen_group=("antigen_group", "first")).reset_index()
    info["n_v1"] = info["complex"].map(cx).fillna(0).astype(int)
    info["fold_complex"] = info["complex"].map(by_complex)
    sizes = np.bincount(by_complex.values, weights=cx[by_complex.index].values, minlength=N_FOLDS)
    for i in info.index[info["fold_complex"].isna()]:
        same = info[(info["antigen_group"] == info.at[i, "antigen_group"]) & info["fold_complex"].notna()]
        info.at[i, "fold_complex"] = same["fold_complex"].iloc[0] if len(same) else int(sizes.argmin())
    info["fold_complex"] = info["fold_complex"].astype(int)

    # secondary split: whole antigen groups held out, balanced on v1 rows
    groups = info["antigen_group"].values
    info["fold_antigen"] = -1
    for k, (_, test) in enumerate(GroupKFold(n_splits=N_FOLDS).split(info, groups=groups)):
        info.loc[info.index[test], "fold_antigen"] = k

    # "seen antigen": another complex of the same antigen sits in a different (i.e. training) fold
    def seen(r):
        o = info[(info["antigen_group"] == r["antigen_group"]) & (info["complex"] != r["complex"])]
        return bool((o["fold_complex"] != r["fold_complex"]).any())

    info["antigen_seen"] = info.apply(seen, axis=1)
    return info, seed


def make_within_complex_folds(dedup_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Reference split: v1 rows of every complex are shuffled and dealt over the folds.

    Every complex appears in train and test ("optimise this exact antibody" scenario), so it is
    easier than the grouped splits; the gap to `fold_complex` measures memorisation of complexes.
    Repeats were already averaged, so the same (complex, mutation) is never on both sides.
    A running counter carries over between complexes so small ones do not all fill fold 0 first.
    """
    v1 = dedup_df[_is_v1(dedup_df)][["complex", "Mutation(s)_cleaned"]].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    v1 = v1.iloc[rng.permutation(len(v1))].sort_values("complex", kind="stable")
    v1["fold_within"] = np.arange(len(v1)) % N_FOLDS
    return v1.sort_index().reset_index(drop=True)
