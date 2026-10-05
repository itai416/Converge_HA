"""Targeted experiments chosen from the error analysis (ridge only, same 5 fold assignments, Huber, unweighted).

H1  antibody-side mutations need their own treatment: geometry x on_antibody interactions
H2  richer 3D context: partner/chain environment (env_*) and wild-type / mutant identity descriptors (wt_*, mut_*)
Stage 1 uses the fast geometry-only ridge; stage 2 repeats the winning variants on top of C (ESM-2 35M embeddings).
Usage: python scripts/12_hypotheses.py [geometry|C]
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import experiments as ex  # noqa: E402
from src.data.io import KEY, PROCESSED, RESULTS  # noqa: E402
from src.data.splits import repeated_complex_folds  # noqa: E402
from src.models.regressors import AugmentedRidge  # noqa: E402
from src.models.run import run_experiments  # noqa: E402

G = ex.GEOMETRY
ENV = ["env_partner_res_8", "env_partner_pos", "env_partner_neg", "env_partner_net_charge", "env_partner_hydrophobic",
       "env_partner_aromatic", "env_same_chain_res_10", "env_same_chain_net_charge", "env_sidechain_contact_frac"]
IDENT = ["wt_hydropathy", "wt_volume", "wt_charge", "wt_aromatic", "wt_gly", "mut_pro", "mut_gly", "mut_hydropathy",
         "mut_volume", "mut_charge"]


def xab(cols):
    return [f"{c}_xab" for c in cols if c != "on_antibody"]


def load(size="35M"):
    dd, df = ex.load(size)
    df = df.merge(pd.read_parquet(PROCESSED / "env_features.parquet"), on=KEY, validate="1:1").reset_index(drop=True)
    for c in G + ENV + IDENT:  # side interactions: the feature's effect may differ between antibody and antigen residues
        if c != "on_antibody":
            df[f"{c}_xab"] = df[c] * df["on_antibody"]
    return dd, df


def variants():
    return {"geometry (reference)": G,
            "H1: geometry + side interactions": G + xab(G),
            "H2a: geometry + environment": G + ENV,
            "H2b: geometry + identity": G + IDENT,
            "H2c: geometry + environment + identity": G + ENV + IDENT,
            "H1+H2: all + side interactions": G + ENV + IDENT + xab(G + ENV + IDENT)}


def main(stage="geometry", n_repeats="5"):
    dd, df = load()
    emb = ex.emb_cols(df)

    def on_C(cols):
        return AugmentedRidge, {"boost_cols": ["llr"], "scalar_cols": cols}, ex.RIDGE_GRID, df[["llr"] + cols + emb]

    if stage == "geometry":
        ref = "geometry (reference): ridge"
        exps = {f"{name}: ridge": (AugmentedRidge, {"scalar_cols": cols}, {"alpha": ex.ALPHAS_LOW_DIM}, df[cols])
                for name, cols in variants().items()}
    else:
        ref = "C: augmented ridge"
        exps = {f"C + {name}: ridge": on_C(cols) for name, cols in variants().items() if not name.startswith("geometry")}
        exps[ref] = on_C(G)
    run_experiments(exps, df, repeated_complex_folds(dd, int(n_repeats)), RESULTS / f"hypotheses_{stage}", n_seeds=1,
                    losses=["huber"], weightings=["none"], pairs={f"{m}  -  {ref}": (m, ref) for m in sorted(exps) if m != ref})


if __name__ == "__main__":
    main(*sys.argv[1:])
