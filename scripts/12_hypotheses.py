"""Targeted experiments chosen from the error analysis (ridge only, same 5 fold assignments, Huber, unweighted).

H1  antibody-side mutations need their own treatment: geometry x on_antibody interactions
H2  richer 3D context: partner/chain environment (env_*) and wild-type / mutant identity descriptors (wt_*, mut_*)
Stage 1 uses the fast geometry-only ridge; stage 2 repeats the winning variants on top of C (ESM-2 35M embeddings).
Usage: python scripts/12_hypotheses.py [geometry|C]
"""
import importlib
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.data.splits import repeated_complex_folds  # noqa: E402
from src.models.regressors import AugmentedRidge  # noqa: E402
from src.models.run import run_experiments  # noqa: E402

abc = importlib.import_module("06_ablation_ABC")
KEY, G = abc.KEY, abc.GEOMETRY
ENV = ["env_partner_res_8", "env_partner_pos", "env_partner_neg", "env_partner_net_charge", "env_partner_hydrophobic",
       "env_partner_aromatic", "env_same_chain_res_10", "env_same_chain_net_charge", "env_sidechain_contact_frac"]
IDENT = ["wt_hydropathy", "wt_volume", "wt_charge", "wt_aromatic", "wt_gly", "mut_pro", "mut_gly", "mut_hydropathy",
         "mut_volume", "mut_charge"]


def load(size="35M"):
    dd, df = abc.load(size)
    env = pd.read_parquet(ROOT / "data" / "processed" / "env_features.parquet")
    df = df.merge(env, on=KEY, validate="1:1").reset_index(drop=True)
    for c in G + ENV + IDENT:  # side interactions: the feature's effect may differ between antibody and antigen residues
        if c != "on_antibody":
            df[f"{c}_xab"] = df[c] * df["on_antibody"]
    return dd, df


def xab(cols):
    return [f"{c}_xab" for c in cols if c != "on_antibody"]


def variants():
    return {"geometry (reference)": G,
            "H1: geometry + side interactions": G + xab(G),
            "H2a: geometry + environment": G + ENV,
            "H2b: geometry + identity": G + IDENT,
            "H2c: geometry + environment + identity": G + ENV + IDENT,
            "H1+H2: all + side interactions": G + ENV + IDENT + xab(G + ENV + IDENT)}


def main(stage="geometry", n_repeats="5"):
    dd, df = load()
    emb = [c for c in df.columns if c.startswith(("wt_emb_", "diff_emb_"))]
    exps = {}
    for name, cols in variants().items():
        if stage == "geometry":
            exps[f"{name}: ridge"] = (AugmentedRidge, {"scalar_cols": cols}, {"alpha": abc.ALPHAS_LOW_DIM}, df[cols])
        elif not name.startswith("geometry"):
            exps[f"C + {name}: ridge"] = (AugmentedRidge, {"boost_cols": ["llr"], "scalar_cols": cols}, abc.RIDGE_GRID,
                                          df[["llr"] + cols + emb])
    if stage != "geometry":
        exps["C: augmented ridge"] = (AugmentedRidge, {"boost_cols": ["llr"], "scalar_cols": G}, abc.RIDGE_GRID,
                                      df[["llr"] + G + emb])
    out = ROOT / "results" / f"hypotheses_{stage}"
    summary = run_experiments(exps, df, repeated_complex_folds(dd, int(n_repeats)), out, n_seeds=1,
                              losses=["huber"], weightings=["none"])
    pd.set_option("display.width", 250)
    print(summary.round(3).to_string())
    r = pd.read_csv(out / "runs.csv").groupby(["model", "repeat"])["per_complex_spearman"].mean().unstack("model")
    ref = "geometry (reference): ridge" if stage == "geometry" else "C: augmented ridge"
    rows = [{"comparison": f"{m}  -  {ref}", "mean_diff": (r[m] - r[ref]).mean(), "min": (r[m] - r[ref]).min(),
             "max": (r[m] - r[ref]).max(), "repeats_better": f"{int(((r[m] - r[ref]) > 0).sum())}/{len(r)}"}
            for m in r.columns if m != ref]
    p = pd.DataFrame(rows).round(3)
    p.to_csv(out / "paired.csv", index=False)
    print("\npaired per-complex Spearman differences (same folds):\n" + p.to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
