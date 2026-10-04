"""Environment + identity features for every v1 row -> data/processed/env_features.parquet"""
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features.environment import residue_environment  # noqa: E402

dd = pd.read_parquet(ROOT / "data" / "processed" / "skempi_abag_dedup.parquet")
v1 = dd[~dd["censored"] & (dd["n_mut"] == 1)]
t0 = time.time()
rows = [{"complex": r["complex"], "Mutation(s)_cleaned": r["Mutation(s)_cleaned"],
         **residue_environment(r["complex"], r["mut_chain"], int(r["file_resnum"]), r["wt_aa"], r["mut_aa"])}
        for _, r in v1.iterrows()]
out = pd.DataFrame(rows)
out.to_parquet(ROOT / "data" / "processed" / "env_features.parquet", index=False)
m = out.merge(v1[["complex", "Mutation(s)_cleaned", "ddG"]], on=["complex", "Mutation(s)_cleaned"])
print(f"{len(out)} rows in {time.time() - t0:.0f}s")
print(m.drop(columns=["complex", "Mutation(s)_cleaned"]).corr("spearman")["ddG"].round(2).to_string())
