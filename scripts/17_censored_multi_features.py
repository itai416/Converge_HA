"""Per-site features for the usable right-censored multi-point rows -> data/processed/censored_multi_sites.parquet

Usable = a numeric lower bound on ddG: mutant affinity ">X" (mut_gt) or wild-type affinity "<X" with an exact mutant (wt_lt).
Same per-site features as 13_multipoint_features.py (geometry + ESM-2 35M), plus the row's ddG_bound.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.mapping import parse_sites  # noqa: E402
from src.features.geometry import residue_features  # noqa: E402
from src.features.sequence import ESM2, chain_sequence  # noqa: E402

KEY = ["complex", "Mutation(s)_cleaned"]
dd = pd.read_parquet(ROOT / "data" / "processed" / "skempi_abag_dedup.parquet")
c = dd[dd["censored"] & (dd["n_mut"] > 1) & dd["censor_type"].isin(["mut_gt", "wt_lt"]) & dd["ddG_bound"].notna()]
c = c.sort_values("ddG_bound").drop_duplicates(KEY, keep="last")
esm = ESM2()
rows = []
for _, r in c.iterrows():
    sites = parse_sites(r)
    assert sites is not None, (r["complex"], r["Mutation(s)_cleaned"])
    for k, (chain, wt, mut, resnum) in enumerate(sites):
        f = residue_features(r["complex"], chain, resnum, wt, mut)
        seq, index = chain_sequence(r["complex"][:4], chain)
        llr, wt_e, diff_e = esm.features(seq, index[resnum], wt, mut)
        rows.append({"complex": r["complex"], "Mutation(s)_cleaned": r["Mutation(s)_cleaned"], "site": k,
                     "ddG_bound": r["ddG_bound"], **f, "llr": llr, **{f"wt_emb_{i}": v for i, v in enumerate(wt_e)},
                     **{f"diff_emb_{i}": v for i, v in enumerate(diff_e)}})
out = pd.DataFrame(rows)
out.to_parquet(ROOT / "data" / "processed" / "censored_multi_sites.parquet", index=False)
print(out.groupby(KEY).ngroups, "censored multi-point rows,", len(out), "sites")
