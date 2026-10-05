"""Chemical environment of the mutated residue in the wild-type complex, and identity descriptors of the substitution (CPU only).

Complements geometry.py (how buried / how connected) with *what* surrounds the site: the partner residues within 8 A of the
side-chain centre, and the packing and charge of the residue's own chain around it.
"""
import numpy as np
from Bio.PDB import NeighborSearch

from src.features.geometry import CHARGE, KD, VOLUME, heavy, partner_search, site

HYDROPHOBIC = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "CYS"}
AROMATIC = {"PHE", "TRP", "TYR", "HIS"}
POS, NEG = {"LYS", "ARG"}, {"ASP", "GLU"}
RES_CHARGE = {"LYS": 1, "ARG": 1, "ASP": -1, "GLU": -1}


def _centre(res):
    """Side-chain centre: CB, or CA for glycine."""
    return (res["CB"] if "CB" in res else res["CA"]).coord


def residue_environment(complex_id, chain, resnum, wt, mut):
    model, res, partner = site(complex_id, chain, resnum)
    centre = _centre(res)
    residues = [r for c in model for r in c if r.id[0] == " " and "CA" in r]
    atoms = [a for r in residues for a in r if a.element != "H"]
    ns = NeighborSearch(atoms)
    near8 = {a.get_parent() for a in ns.search(centre, 8.0)}
    part = [r for r in near8 if r.get_parent().id in partner]
    same = [r for r in {a.get_parent() for a in ns.search(centre, 10.0)}
            if r.get_parent().id == chain and r is not res]
    same8 = [r for r in near8 if r.get_parent().id == chain and r is not res]
    names = [r.get_resname() for r in part]
    # fraction of side-chain heavy atoms (beyond the backbone N, CA, C, O) within 4.5 A of a partner atom
    pns = partner_search(model, partner)
    side = [a for a in heavy(res) if a.get_id() not in ("N", "CA", "C", "O")] or [res["CA"]]
    sc_contact = np.mean([bool(pns.search(a.coord, 4.5)) for a in side])
    f = {"env_partner_res_8": len(part),
         "env_partner_pos": sum(n in POS for n in names), "env_partner_neg": sum(n in NEG for n in names),
         "env_partner_net_charge": sum(RES_CHARGE.get(n, 0) for n in names),
         "env_partner_hydrophobic": sum(n in HYDROPHOBIC for n in names),
         "env_partner_aromatic": sum(n in AROMATIC for n in names),
         "env_same_chain_res_10": len(same),
         "env_same_chain_net_charge": sum(RES_CHARGE.get(r.get_resname(), 0) for r in same8),
         "env_sidechain_contact_frac": float(sc_contact)}
    f.update({"wt_hydropathy": KD[wt], "wt_volume": VOLUME[wt], "wt_charge": CHARGE.get(wt, 0),
              "wt_aromatic": int(wt in "FWYH"), "wt_gly": int(wt == "G"),
              "mut_pro": int(mut == "P"), "mut_gly": int(mut == "G"), "mut_hydropathy": KD[mut], "mut_volume": VOLUME[mut],
              "mut_charge": CHARGE.get(mut, 0)})
    return f
