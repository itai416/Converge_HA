"""Stage 4 (structure, learned): ESM-IF1 features for every v1 row -> data/processed/esmif1.parquet

Scalar scores (site and whole-chain log-likelihood ratios, in complex / unbound / their difference) plus the site encoder
embedding in both contexts. See src/features/inverse_folding.py. Meant for a GPU (Colab); runs on CPU too, slowly.
Usage: python scripts/08_esmif1.py [--limit N]   (--limit: only the first N rows, for a quick test)

Progress is checkpointed every CHECKPOINT_EVERY mutations; rerunning the same command resumes where it stopped
(useful when a Colab session disconnects, or on CPU where the full run takes ~2.5 h).
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features.inverse_folding import ESMIF1  # noqa: E402

KEY = ["complex", "Mutation(s)_cleaned"]
CHECKPOINT_EVERY = 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    dd = pd.read_parquet(ROOT / "data" / "processed" / "skempi_abag_dedup.parquet")
    v1 = dd[~dd["censored"] & (dd["n_mut"] == 1)].reset_index(drop=True)
    if args.limit:
        v1 = v1.head(args.limit)
    model = ESMIF1()
    path = ROOT / "data" / "processed" / ("esmif1.parquet" if not args.limit else f"esmif1_test{args.limit}.parquet")
    ckpt = path.with_suffix(".partial.pkl")
    done = pickle.loads(ckpt.read_bytes()) if ckpt.exists() else {}  # (complex, mutation) -> (features, emb_c, emb_u)
    print(f"ESM-IF1 on {model.device}, {len(v1)} mutations, {len(done)} already done (resuming)", flush=True)

    for i, r in v1.iterrows():
        key = (r["complex"], r["Mutation(s)_cleaned"])
        if key in done:
            continue
        f, emb = model.features(r["complex"], r["mut_chain"], int(r["file_resnum"]), r["wt_aa"], r["mut_aa"])
        done[key] = (f, emb["complex"], emb["unbound"])
        if len(done) % CHECKPOINT_EVERY == 0:
            ckpt.write_bytes(pickle.dumps(done))
            print(f"  {len(done)}/{len(v1)}  {time.time() - t0:.0f}s (checkpoint saved)", flush=True)

    keys = list(zip(v1["complex"], v1["Mutation(s)_cleaned"]))
    rows = [{"complex": k[0], "Mutation(s)_cleaned": k[1], **done[k][0]} for k in keys]
    emb_c, emb_u = np.stack([done[k][1] for k in keys]), np.stack([done[k][2] for k in keys])
    d = emb_c.shape[1]
    out = pd.concat([pd.DataFrame(rows),
                     pd.DataFrame(emb_c, columns=[f"if_emb_complex_{j}" for j in range(d)]),
                     pd.DataFrame(emb_u, columns=[f"if_emb_unbound_{j}" for j in range(d)])], axis=1)
    out.to_parquet(path, index=False)
    ckpt.unlink(missing_ok=True)
    print(f"{len(out)} rows, embedding dim {d}, {time.time() - t0:.0f}s -> {path}")

    m = out.merge(v1[KEY + ["ddG"]], on=KEY)
    scores = [c for c in out.columns if c.startswith(("if_site", "if_seq"))]
    print("Spearman with ddG (expect negative: a less likely mutant should bind worse):")
    print(m[scores + ["ddG"]].corr("spearman")["ddG"].drop("ddG").round(3).to_string())


if __name__ == "__main__":
    main()
