"""The three agreed metrics, with bootstrap confidence intervals over complexes.

per_complex_spearman  mean Spearman over complexes with >= MIN_ROWS test rows (primary: ranking within one antibody)
pooled_pearson        Pearson over all test rows (comparable with published SKEMPI numbers)
rmse                  kcal/mol (are the predicted values the right size?)
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr

MIN_ROWS = 10
warnings.filterwarnings("ignore", category=ConstantInputWarning)  # constant predictions -> NaN, handled below


def _per_complex(df):
    r = [spearmanr(g["pred"], g["ddG"])[0] for _, g in df.groupby("complex") if len(g) >= MIN_ROWS]
    r = [x for x in r if np.isfinite(x)]  # a constant prediction gives NaN
    return float(np.mean(r)) if r else np.nan


def metrics(df: pd.DataFrame) -> dict:
    """df has columns complex, ddG, pred (out-of-fold predictions)."""
    pooled = pearsonr(df["pred"], df["ddG"])[0] if df["pred"].std() > 0 else np.nan
    return {"per_complex_spearman": _per_complex(df), "pooled_pearson": pooled,
            "rmse": float(np.sqrt(((df["pred"] - df["ddG"]) ** 2).mean()))}


def bootstrap(df: pd.DataFrame, n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Point estimate and 95% CI per metric, resampling whole complexes with replacement."""
    rng = np.random.default_rng(seed)
    groups = {c: g for c, g in df.groupby("complex")}
    names = list(groups)
    reps = []
    for _ in range(n):
        pick = rng.choice(names, size=len(names), replace=True)
        # relabel so a complex drawn twice counts as two complexes in the per-complex mean
        reps.append(metrics(pd.concat([groups[c].assign(complex=f"{c}#{i}") for i, c in enumerate(pick)])))
    reps = pd.DataFrame(reps)
    point = metrics(df)
    return pd.DataFrame({m: {"value": point[m], "lo": reps[m].quantile(0.025), "hi": reps[m].quantile(0.975)}
                         for m in point}).T
