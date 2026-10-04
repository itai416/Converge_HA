"""Per-site features (geometry + ESM-2) for every site of the uncensored multi-point rows -> data/processed/multi_sites.parquet

One row per (multi-point mutation, site). Features are those of the single-point model C, computed on the wild-type
structure and wild-type chain for each site independently (the other mutations of the set are not applied).
Usage: python scripts/13_multipoint_features.py [hf_model_name]   (default ESM-2 35M)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.mapping import parse_sites  # noqa: E402
from src.features.geometry import residue_features  # noqa: E402
from src.features.sequence import ESM2, chain_sequence  # noqa: E402


def main(name="facebook/esm2_t12_35M_UR50D"):
    t0 = time.time()
    dd = pd.read_parquet(ROOT / "data" / "processed" / "skempi_abag_dedup.parquet")
    multi = dd[~dd["censored"] & (dd["n_mut"] > 1)]
    esm = ESM2(name)
    rows, dropped = [], []
    for _, r in multi.iterrows():
        sites = parse_sites(r)
        if sites is None:
            dropped.append((r["complex"], r["Mutation(s)_cleaned"], "site not located / wild-type mismatch"))
            continue
        try:
            feats = []
            for k, (chain, wt, mut, resnum) in enumerate(sites):
                f = residue_features(r["complex"], chain, resnum, wt, mut)
                seq, index = chain_sequence(r["complex"][:4], chain)
                llr, wt_e, diff_e = esm.features(seq, index[resnum], wt, mut)
                feats.append({"complex": r["complex"], "Mutation(s)_cleaned": r["Mutation(s)_cleaned"], "site": k, **f,
                              "llr": llr, **{f"wt_emb_{i}": v for i, v in enumerate(wt_e)},
                              **{f"diff_emb_{i}": v for i, v in enumerate(diff_e)}})
            rows += feats
        except (StopIteration, KeyError, AssertionError) as e:  # residue missing from the file / not in the chain sequence
            dropped.append((r["complex"], r["Mutation(s)_cleaned"], repr(e)))
    out = pd.DataFrame(rows)
    out.to_parquet(ROOT / "data" / "processed" / "multi_sites.parquet", index=False)
    kept = out.groupby(["complex", "Mutation(s)_cleaned"]).ngroups
    print(f"{len(multi)} uncensored multi-point rows: {kept} kept ({len(out)} sites), {len(dropped)} dropped, {time.time() - t0:.0f}s")
    for d in dropped:
        print("dropped:", *d)


if __name__ == "__main__":
    main(*sys.argv[1:])
