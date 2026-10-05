"""Stage 1, step 1: build data/processed/skempi_abag.parquet (one row per SKEMPI AB/AG measurement)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.clean import load_clean  # noqa: E402
from src.data.dedup import dedup  # noqa: E402
from src.data.io import PROCESSED, is_v1  # noqa: E402

OUT = PROCESSED / "skempi_abag.parquet"
OUT_DEDUP = PROCESSED / "skempi_abag_dedup.parquet"


def main():
    df = load_clean()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    v1 = df[is_v1(df)]
    print(f"rows: {len(df)}, complexes: {df.complex.nunique()}")
    print(f"censored: {df.censored.sum()}")
    print(df["censor_type"].value_counts().to_string())
    print("n_mut distribution:")
    print(df["n_mut"].value_counts().sort_index().to_string())
    print(f"v1 (~censored & n_mut == 1): {len(v1)} rows in {v1.complex.nunique()} complexes")
    print(f"saved to {OUT}")

    dd = dedup(df)
    dd.to_parquet(OUT_DEDUP, index=False)
    v1d = dd[is_v1(dd)]
    rep = v1d[v1d["n_repeats"] > 1]
    print(f"dedup: {len(df)} -> {len(dd)} rows; v1 {len(v1)} -> {len(v1d)}")
    print(f"v1 repeated mutations: {len(rep)}, ddG std within repeats: "
          f"median {rep['ddG_std'].median():.2f}, max {rep['ddG_std'].max():.2f}")
    print(f"saved to {OUT_DEDUP}")


if __name__ == "__main__":
    main()
