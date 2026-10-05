"""Environment + identity features for every v1 row -> data/processed/env_features.parquet"""
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.io import KEY, PROCESSED, is_v1, load_dedup  # noqa: E402
from src.features.environment import residue_environment  # noqa: E402


def main():
    t0 = time.time()
    dd = load_dedup()
    v1 = dd[is_v1(dd)]
    out = pd.DataFrame([{**r[KEY], **residue_environment(r["complex"], r["mut_chain"], int(r["file_resnum"]), r["wt_aa"], r["mut_aa"])}
                        for _, r in v1.iterrows()])
    out.to_parquet(PROCESSED / "env_features.parquet", index=False)
    m = out.merge(v1[KEY + ["ddG"]], on=KEY)
    print(f"{len(out)} rows in {time.time() - t0:.0f}s")
    print(m.drop(columns=KEY).corr("spearman")["ddG"].round(2).to_string())


if __name__ == "__main__":
    main()
