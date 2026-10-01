"""Stage 4 (CPU part): geometric features of the mutated site for every v1 row -> data/processed/geom_features.parquet"""
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features.geometry import residue_features  # noqa: E402


def main():
    t0 = time.time()
    dd = pd.read_parquet(ROOT / "data" / "processed" / "skempi_abag_dedup.parquet")
    v1 = dd[~dd["censored"] & (dd["n_mut"] == 1)].copy()
    rows = []
    for _, r in v1.iterrows():
        f = residue_features(r["complex"], r["mut_chain"], int(r["file_resnum"]), r["wt_aa"], r["mut_aa"])
        rows.append({"complex": r["complex"], "Mutation(s)_cleaned": r["Mutation(s)_cleaned"], **f})
    out = pd.DataFrame(rows)
    out.to_parquet(ROOT / "data" / "processed" / "geom_features.parquet", index=False)
    print(f"{len(out)} rows in {time.time() - t0:.0f}s")
    print(out.describe().T[["mean", "std", "min", "max"]].round(2).to_string())
    m = out.merge(v1[["complex", "Mutation(s)_cleaned", "ddG"]], on=["complex", "Mutation(s)_cleaned"])
    print("\nSpearman with ddG:\n", m.drop(columns=["complex", "Mutation(s)_cleaned"]).corr("spearman")["ddG"].round(2).to_string())


if __name__ == "__main__":
    main()
