"""Average repeated measurements of the same (complex, mutation)."""
import pandas as pd

from src.data.io import KEY


def dedup(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (complex, mutation) among exact rows; censored rows pass through unchanged.

    Adds `ddG_std` (std of the repeats, NaN when there is one measurement) and `n_repeats`.
    Averaging is done in ddG space. Other columns come from the first repeat (for exact rows
    these agree within a group, except Reference/Method/T, which can differ between papers).
    """
    exact, cens = df[~df["censored"]], df[df["censored"]]
    g = exact.groupby(KEY, sort=False)
    out = g.first()
    out["ddG"] = g["ddG"].mean()
    out["ddG_std"] = g["ddG"].std()
    out["n_repeats"] = g.size()
    out = out.reset_index()[list(df.columns) + ["ddG_std", "n_repeats"]]
    cens = cens.assign(ddG_std=float("nan"), n_repeats=1)
    return pd.concat([out, cens], ignore_index=True)
