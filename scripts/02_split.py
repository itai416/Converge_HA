"""Stage 3: write data/processed/folds.csv (one row per complex) and print a balance report."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.io import KEY, PROCESSED, is_v1, load_dedup  # noqa: E402
from src.data.splits import make_folds, make_within_complex_folds  # noqa: E402

OUT = PROCESSED / "folds.csv"
OUT_WITHIN = PROCESSED / "folds_within_complex.csv"


def report(v1, info, col):
    t = v1.merge(info[["complex", col]], on="complex")
    g = t.groupby(col)
    r = pd.DataFrame({
        "complexes": info.groupby(col).size(), "v1_rows": g.size(),
        "antigens": g["antigen_group"].nunique(), "ddG_mean": g["ddG"].mean().round(2),
        "ddG_std": g["ddG"].std().round(2), "pct_improving": (g["ddG"].apply(lambda s: (s < -0.5).mean()) * 100).round(1),
    })
    print(f"\n{col}\n{r.to_string()}")


def main():
    dd = load_dedup()
    info, seed = make_folds(dd)
    info.to_csv(OUT, index=False)
    v1 = dd[is_v1(dd)]
    print(f"seed {seed}, {len(info)} complexes, {len(v1)} v1 rows, saved to {OUT}")
    report(v1, info, "fold_complex")
    report(v1, info, "fold_antigen")
    t = v1.merge(info[["complex", "fold_complex", "antigen_seen"]], on="complex")
    print("\nfold_complex: test rows with seen antigen:", f"{t['antigen_seen'].mean():.0%}")
    print(t.groupby("fold_complex")["antigen_seen"].mean().round(2).to_dict())
    lys = info[info["antigen_group"] == "lysozyme"]
    print("lysozyme complexes per fold_complex:", lys["fold_complex"].value_counts().sort_index().to_dict())

    w = make_within_complex_folds(dd)
    w.to_csv(OUT_WITHIN, index=False)
    t = v1.merge(w, on=KEY)
    assert len(t) == len(v1)
    g = t.groupby("fold_within")
    print(f"\nfold_within (saved to {OUT_WITHIN})")
    print(pd.DataFrame({"v1_rows": g.size(), "complexes": g["complex"].nunique(),
                        "ddG_mean": g["ddG"].mean().round(2), "ddG_std": g["ddG"].std().round(2)}).to_string())


if __name__ == "__main__":
    main()
