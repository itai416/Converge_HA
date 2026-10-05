"""Residue mapping: locate each single-point mutation in the PDB structure and check the wild-type."""
import re
from functools import lru_cache

import pandas as pd

from src.data.io import ROOT

PDB_DIR = ROOT / "data" / "SKEMPI2_PDBs" / "PDBs"

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
MUT = re.compile(r"^([A-Z])([A-Za-z0-9])(-?\d+[A-Za-z]?)([A-Z])$")  # e.g. LI45G, RD100bA


@lru_cache(maxsize=None)
def read_mapping(pdb_id: str) -> dict:
    """{(chain, 'resnum+icode'): (one-letter wt residue, residue number in the provided .pdb file (renumbered by SKEMPI))} from <PDB>.mapping.

    Fixed-width lines: resname [0:3], chain [4], resnum [5:9], insertion code [9], seq index [10:].
    """
    out = {}
    for line in (PDB_DIR / f"{pdb_id}.mapping").read_text().splitlines():
        if not line.strip():
            continue
        res3, chain = line[0:3], line[4]
        resnum = line[5:9].strip() + line[9].strip().upper()
        out[(chain, resnum)] = (AA3.get(res3, "X"), int(line[10:]))
    return out


def _locate(complex_id: str, token: str):
    """One mutation token -> (chain, wt, mut, PDB residue number, mapping entry or None); None if the token does not parse."""
    m = MUT.match(token)
    if not m:
        return None
    wt, chain, resnum, mut = m.groups()
    return chain, wt, mut, resnum, read_mapping(complex_id[:4]).get((chain, resnum.upper()))


def add_residue_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Add mutation-parsing and PDB lookup columns. Single-point rows only; multi-point rows stay NaN."""
    cols = ["mut_chain", "wt_aa", "mut_aa", "pdb_resnum", "file_resnum", "pdb_wt_aa"]
    df = df.assign(**{c: pd.NA for c in cols}, wt_match=pd.NA)
    for i, row in df[df["n_mut"] == 1].iterrows():
        chain, wt, mut, resnum, hit = _locate(row["complex"], row["Mutation(s)_PDB"])
        df.loc[i, ["mut_chain", "wt_aa", "mut_aa", "pdb_resnum"]] = [chain, wt, mut, resnum]
        if hit:
            df.loc[i, ["pdb_wt_aa", "file_resnum"]] = hit
        df.loc[i, "wt_match"] = bool(hit) and hit[0] == wt
    df["wt_match"] = df["wt_match"].astype("boolean")
    return df


def parse_sites(row) -> list:
    """All sites of a (single- or multi-point) row as (chain, wt, mut, file residue number); None if any site cannot be located
    or its wild-type residue disagrees with the structure."""
    sites = []
    for token in row["Mutation(s)_PDB"].split(","):
        found = _locate(row["complex"], token)
        if found is None:
            return None
        chain, wt, mut, _, hit = found
        if not hit or hit[0] != wt:
            return None
        sites.append((chain, wt, mut, hit[1]))
    return sites
