"""Pack what the Colab notebook needs into colab_bundle.zip (~10 MB): code, the cleaned table and the AB/AG structures.

Upload the zip to Google Drive (default location: MyDrive/converge_ha/colab_bundle.zip) and run
notebooks/02_colab_embeddings.ipynb. Not needed once the repository is on GitHub (the notebook can clone it instead,
but the SKEMPI structures are not in the repository and still come from this bundle or from the SKEMPI website).
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.io import PROCESSED, load_dedup  # noqa: E402
from src.data.mapping import PDB_DIR  # noqa: E402

OUT = ROOT / "colab_bundle.zip"


def main():
    pdb_ids = sorted({c[:4] for c in load_dedup()["complex"]})
    files = list((ROOT / "src").rglob("*.py"))
    files += [ROOT / "scripts" / n for n in ["04_esm2.py", "08_esmif1.py"]]
    files += [PROCESSED / "skempi_abag_dedup.parquet"]
    files += [PDB_DIR / f"{pid}.{ext}" for pid in pdb_ids for ext in ("pdb", "mapping")]
    missing = [f for f in files if not f.exists()]
    assert not missing, missing
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(ROOT).as_posix())
    print(f"{len(files)} files ({len(pdb_ids)} structures) -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
