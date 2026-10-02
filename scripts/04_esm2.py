"""Stage 4 (sequence): ESM-2 site features for every v1 row -> data/processed/esm2_<size>.parquet

Usage: python scripts/04_esm2.py [hf_model_name]   (default ESM-2 35M; on Colab pass facebook/esm2_t33_650M_UR50D)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features.sequence import ESM2, chain_sequence  # noqa: E402


def main(name="facebook/esm2_t12_35M_UR50D"):
    t0 = time.time()
    dd = pd.read_parquet(ROOT / "data" / "processed" / "skempi_abag_dedup.parquet")
    v1 = dd[~dd["censored"] & (dd["n_mut"] == 1)]
    esm = ESM2(name)
    keys, llr, wt, diff = [], [], [], []
    for _, r in v1.iterrows():
        seq, index = chain_sequence(r["complex"][:4], r["mut_chain"])
        a, b, c = esm.features(seq, index[int(r["file_resnum"])], r["wt_aa"], r["mut_aa"])
        keys.append((r["complex"], r["Mutation(s)_cleaned"]))
        llr.append(a), wt.append(b), diff.append(c)
    d = np.stack(wt).shape[1]
    out = pd.concat([pd.DataFrame(keys, columns=["complex", "Mutation(s)_cleaned"]),
                     pd.DataFrame({"llr": llr}),
                     pd.DataFrame(np.stack(wt), columns=[f"wt_emb_{i}" for i in range(d)]),
                     pd.DataFrame(np.stack(diff), columns=[f"diff_emb_{i}" for i in range(d)])], axis=1)
    size = name.split("_")[-2]
    path = ROOT / "data" / "processed" / f"esm2_{size}.parquet"
    out.to_parquet(path, index=False)
    m = out.merge(v1[["complex", "Mutation(s)_cleaned", "ddG"]], on=["complex", "Mutation(s)_cleaned"])
    print(f"{len(out)} rows, dim {d}, {time.time() - t0:.0f}s -> {path}")
    print(f"zero-shot Spearman(-llr, ddG) = {(-m['llr']).corr(m['ddG'], method='spearman'):.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
