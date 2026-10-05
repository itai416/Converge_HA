"""Stage 4 (CPU part): geometric features of the mutated site for every v1 row -> data/processed/geom_features.parquet"""
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.io import KEY, PROCESSED, is_v1, load_dedup  # noqa: E402
from src.features.geometry import residue_features  # noqa: E402


def main():
    t0 = time.time()
    dd = load_dedup()
    v1 = dd[is_v1(dd)]
    out = pd.DataFrame([{**r[KEY], **residue_features(r["complex"], r["mut_chain"], int(r["file_resnum"]), r["wt_aa"], r["mut_aa"])}
                        for _, r in v1.iterrows()])
    out.to_parquet(PROCESSED / "geom_features.parquet", index=False)
    print(f"{len(out)} rows in {time.time() - t0:.0f}s")
    print(out.describe().T[["mean", "std", "min", "max"]].round(2).to_string())
    m = out.merge(v1[KEY + ["ddG"]], on=KEY)
    print("\nSpearman with ddG:\n", m.drop(columns=KEY).corr("spearman")["ddG"].round(2).to_string())


if __name__ == "__main__":
    main()
