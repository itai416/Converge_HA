"""Step 1 of Stage 1: filter SKEMPI 2.0 to AB/AG rows and build the label columns."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.antigen_groups import ANTIGEN_GROUP
from src.data.mapping import add_residue_mapping

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "skempi_v2.csv"
R = 0.0019872041  # kcal / (mol K)
DEFAULT_T = 298.0


def _bound(raw: pd.Series, side: str) -> pd.Series:
    """'mut_gt' / 'mut_lt' / 'nb' / '' for one affinity column."""
    s = raw.astype(str).str.strip()
    out = pd.Series("", index=raw.index)
    out[s.str.startswith(">")] = f"{side}_gt"
    out[s.str.startswith("<")] = f"{side}_lt"
    out[s.str.startswith("n.b")] = "nb"
    return out


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(CSV, sep=";")
    df = df[df["Hold_out_type"].fillna("").str.contains("AB/AG")].copy()
    df = df.rename(columns={"#Pdb": "complex"}).reset_index(drop=True)

    missing = set(df["complex"]) - set(ANTIGEN_GROUP)
    assert not missing, f"complexes without antigen group: {sorted(missing)}"
    df["antigen_group"] = df["complex"].map(ANTIGEN_GROUP)

    df["n_mut"] = df["Mutation(s)_cleaned"].str.count(",") + 1

    temp = pd.to_numeric(df["Temperature"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    df["T"] = temp.fillna(DEFAULT_T)

    types = pd.concat([_bound(df["Affinity_mut (M)"], "mut"), _bound(df["Affinity_wt (M)"], "wt")], axis=1)
    df["censor_type"] = types.apply(lambda r: "+".join(t for t in r if t) or "none", axis=1)
    df["censored"] = df["censor_type"] != "none"

    ratio = np.log(df["Affinity_mut_parsed"] / df["Affinity_wt_parsed"])
    value = R * df["T"] * ratio
    df["ddG"] = value.where(~df["censored"])
    df["ddG_bound"] = value.where(df["censored"])  # NaN for n.b. (no parsed affinity)
    return add_residue_mapping(df)
