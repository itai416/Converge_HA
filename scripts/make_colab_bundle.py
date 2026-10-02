"""Pack what the Colab notebook needs into colab_bundle.zip (~10 MB): code, the cleaned table and the AB/AG structures.

Upload the zip to Google Drive (default location: MyDrive/converge_ha/colab_bundle.zip) and run
notebooks/02_colab_embeddings.ipynb. Not needed once the repository is on GitHub (the notebook can clone it instead,
but the SKEMPI structures are not in the repository and still come from this bundle or from the SKEMPI website).
"""
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "colab_bundle.zip"


def main():
    dd = pd.read_parquet(ROOT / "data" / "processed" / "skempi_abag_dedup.parquet")
    pdb_ids = sorted({c[:4] for c in dd["complex"]})
    files = [p for p in (ROOT / "src").rglob("*.py")]
    files += [ROOT / "scripts" / n for n in ["04_esm2.py", "08_esmif1.py"]]
    files += [ROOT / "data" / "processed" / "skempi_abag_dedup.parquet"]
    for pid in pdb_ids:
        files += [ROOT / "data" / "SKEMPI2_PDBs" / "PDBs" / f"{pid}.{ext}" for ext in ("pdb", "mapping")]
    missing = [f for f in files if not f.exists()]
    assert not missing, missing
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(ROOT).as_posix())
    print(f"{len(files)} files ({len(pdb_ids)} structures) -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
