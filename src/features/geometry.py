"""Hand-crafted geometric features of the mutated residue in the wild-type complex (CPU only)."""
import warnings
from functools import lru_cache

import numpy as np
from Bio.PDB import NeighborSearch, PDBParser
from Bio.PDB.SASA import ShrakeRupley
from Bio.Align import substitution_matrices

from src.data.mapping import PDB_DIR

# Tien et al. 2013 (theoretical) maximum accessible surface area, A^2
MAX_ASA = {"ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167, "GLN": 225, "GLU": 223, "GLY": 104,
           "HIS": 224, "ILE": 197, "LEU": 201, "LYS": 236, "MET": 224, "PHE": 240, "PRO": 159, "SER": 155,
           "THR": 172, "TRP": 285, "TYR": 263, "VAL": 174}
VOLUME = dict(zip("ACDEFGHIKLMNPQRSTVWY", [88.6, 108.5, 111.1, 138.4, 189.9, 60.1, 153.2, 166.7, 168.6, 166.7,
                                          162.9, 114.1, 112.7, 143.8, 173.4, 89.0, 116.1, 140.0, 227.8, 193.6]))
KD = dict(zip("ACDEFGHIKLMNPQRSTVWY", [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8, 1.9, -3.5, -1.6,
                                      -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]))
CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1}
POLAR = {"N", "O"}
CHARGED_ATOMS = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2"),
                 ("LYS", "NZ"), ("ARG", "NH1"), ("ARG", "NH2"), ("ARG", "NE")}
BLOSUM = substitution_matrices.load("BLOSUM62")

# partner that is the antibody (partner 1 = first chain group in the #Pdb id), from the SKEMPI protein names
AB_IS_PARTNER2 = {"1DVF_AB_CD": False, "1NCA_N_LH": True, "3BN9_B_CD": True, "3NPS_A_BC": True,
                  "4GXU_ABCDEF_MN": True, "3LZF_AB_HL": True, "3N85_A_LH": True, "3L5X_A_HL": True,
                  "2BDN_HL_A": False, "3W2D_A_HL": True, "4KRL_A_B": True, "4KRO_A_B": True, "4KRP_A_B": True,
                  "4NM8_ABCDEF_HL": True, "1JRH_LH_I": False, "2JEL_LH_P": False, "1NMB_N_LH": True,
                  "5DWU_HL_AB": False}


def antibody_chains(complex_id: str):
    """Return (antibody chains, antigen chains) for a SKEMPI '#Pdb' id such as 1VFB_AB_C."""
    _, p1, p2 = complex_id.split("_")
    ab_second = AB_IS_PARTNER2.get(complex_id, False)
    return (p2, p1) if ab_second else (p1, p2)


@lru_cache(maxsize=None)
def _structure(pdb_id):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = PDBParser(QUIET=True).get_structure(pdb_id, str(PDB_DIR / f"{pdb_id}.pdb"))
    return s[0]


def _res_sasa(model, chains):
    """Residue SASA computed with only `chains` present."""
    from Bio.PDB.Model import Model
    sub = Model(0)
    for c in chains:
        if c in model:
            sub.add(model[c].copy())
    ShrakeRupley().compute(sub, level="R")
    return {(c.id, r.id): r.sasa for c in sub for r in c}


@lru_cache(maxsize=None)
def _sasa_tables(complex_id):
    model = _structure(complex_id[:4])
    ab, ag = antibody_chains(complex_id)
    return _res_sasa(model, ab + ag), _res_sasa(model, ab), _res_sasa(model, ag)


def residue_features(complex_id, chain, resnum, wt, mut):
    model = _structure(complex_id[:4])
    ab, ag = antibody_chains(complex_id)
    on_ab = chain in ab
    res = next(r for r in model[chain] if r.id[1] == resnum and r.id[0] == " ")
    key = (chain, res.id)
    cx, abo, ago = _sasa_tables(complex_id)
    unbound = (abo if on_ab else ago)[key]
    partner = set(ag if on_ab else ab)
    partner_atoms = [a for c in partner if c in model for a in model[c].get_atoms() if a.element != "H"]
    ns = NeighborSearch(partner_atoms)
    atoms = [a for a in res if a.element != "H"]
    f = {"on_antibody": int(on_ab), "rsa_complex": cx[key] / MAX_ASA[res.get_resname()],
         "rsa_unbound": unbound / MAX_ASA[res.get_resname()], "dsasa": unbound - cx[key]}
    for cut in (4.5, 8.0):
        near = {id(p) for a in atoms for p in ns.search(a.coord, cut)}
        f[f"partner_atoms_{int(cut) if cut == 8.0 else cut}"] = len(near)
    d = [np.linalg.norm(a.coord - p.coord) for a in atoms for p in ns.search(a.coord, 12.0)]
    f["min_dist_partner"] = min(d) if d else 12.0
    hb = sb = 0
    for a in atoms:
        if a.element in POLAR:
            for p in ns.search(a.coord, 3.5):
                if p.element in POLAR:
                    hb += 1
                    if (res.get_resname(), a.get_id()) in CHARGED_ATOMS and \
                            (p.get_parent().get_resname(), p.get_id()) in CHARGED_ATOMS and \
                            CHARGE.get(wt) is not None:
                        sb += 1
    f["cross_hbonds"], f["cross_saltbridges"] = hb, sb
    f["bfactor"] = float(np.mean([a.bfactor for a in atoms]))
    f["d_volume"], f["d_hydropathy"] = VOLUME[mut] - VOLUME[wt], KD[mut] - KD[wt]
    f["d_charge"] = CHARGE.get(mut, 0) - CHARGE.get(wt, 0)
    f["blosum62"] = float(BLOSUM[wt][mut])
    return f
