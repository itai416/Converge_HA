"""Features for the usable censored single-point rows -> data/processed/censored_features.parquet

Usable = right-censored with a numeric bound: the mutant affinity is ">X" (mut_gt) or the wild-type affinity is "<X" with an exact
mutant affinity (wt_lt). In both cases the true ddG is >= `ddG_bound`. Rows where both affinities are "<" (no direction), non-binders
without a number, and the 5 multi-point upper bounds are not used. Repeated (complex, mutation) keys keep the largest bound.
Same per-site features as model C (geometry + ESM-2 llr and embeddings).
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
c = dd[dd["censored"] & (dd["n_mut"] == 1) & dd["censor_type"].isin(["mut_gt", "wt_lt"]) & dd["ddG_bound"].notna()]
c = c.sort_values("ddG_bound").drop_duplicates(KEY, keep="last")
esm = ESM2()
rows = []
for _, r in c.iterrows():
    (chain, wt, mut, resnum), = parse_sites(r)
    f = residue_features(r["complex"], chain, resnum, wt, mut)
    seq, index = chain_sequence(r["complex"][:4], chain)
    llr, wt_e, diff_e = esm.features(seq, index[resnum], wt, mut)
    rows.append({"complex": r["complex"], "Mutation(s)_cleaned": r["Mutation(s)_cleaned"], "ddG_bound": r["ddG_bound"],
                 "censor_type": r["censor_type"], **f, "llr": llr, **{f"wt_emb_{i}": v for i, v in enumerate(wt_e)},
                 **{f"diff_emb_{i}": v for i, v in enumerate(diff_e)}})
out = pd.DataFrame(rows)
out.to_parquet(ROOT / "data" / "processed" / "censored_features.parquet", index=False)
print(len(out), "rows,", out["complex"].nunique(), "complexes;", out["censor_type"].value_counts().to_dict())
print(out["ddG_bound"].describe().round(2).to_dict())
