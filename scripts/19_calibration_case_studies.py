"""Stage 8, second part: calibration by bin and structural case studies of the worst errors, for model C.

Uses the same out-of-fold predictions as 10_error_analysis.py (primary split, repeat 0). C augmented ridge is the model studied;
C LightGBM is drawn in the calibration plot for comparison. Writes to results/error_analysis/:
  calibration_by_pred_bin.csv, calibration_summary.csv, calibration.png
  case_studies.csv, case_contacts.csv, case_same_site.csv, case_studies.png, pymol/<complex>_<mutation>.pml
Usage: python scripts/19_calibration_case_studies.py
"""
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from Bio.PDB import NeighborSearch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.io import KEY, RESULTS, load_oof_frame  # noqa: E402
from src.data.mapping import AA3, read_mapping  # noqa: E402
from src.eval.metrics import resample_complexes  # noqa: E402
from src.features.geometry import CHARGED_ATOMS, POLAR, antibody_chains, heavy, site as locate_site  # noqa: E402

OUT = RESULTS / "error_analysis"
(OUT / "pymol").mkdir(parents=True, exist_ok=True)
MODELS = {"C: ridge": "C: augmented ridge | huber | w=none", "C: LightGBM": "C: LightGBM | huber | w=none"}
MAIN = "C: ridge"
N_BINS, N_BOOT, N_CASES, N_ANALOGUES = 10, 1000, 12, 20
STRUCT = ["rsa_complex", "rsa_unbound", "dsasa", "partner_atoms_4.5", "partner_atoms_8", "min_dist_partner", "cross_hbonds",
          "cross_saltbridges", "bfactor"]
SUBST = ["d_volume", "d_hydropathy", "d_charge", "blosum62"]
BACKBONE = {"N", "CA", "C", "O", "OXT"}
RINGS = {"PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"], "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
         "TRP": ["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"], "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"]}
CATIONS = {("LYS", "NZ"), ("ARG", "CZ")}

oof, base = load_oof_frame(["iMutation_Location(s)", "antigen_group", "mut_chain", "wt_aa", "mut_aa", "pdb_resnum", "file_resnum",
                            "Affinity_wt_parsed", "Affinity_mut_parsed", "Method", "Notes", "ddG_std", "n_repeats"])
for name, col in MODELS.items():
    base[name] = oof[col].values


# ---------------------------------------------------------------- calibration by bin
def by_pred_bin(d, pred, edges):
    """Mean prediction and observed ddG per bin of the prediction (fixed edges)."""
    b = np.clip(np.searchsorted(edges, d[pred].values, side="right") - 1, 0, len(edges) - 2)
    return d.assign(bin=b).groupby("bin").agg(rows=("ddG", "size"), mean_pred=(pred, "mean"), mean_obs=("ddG", "mean"),
                                              obs_q10=("ddG", lambda x: x.quantile(.1)), obs_q50=("ddG", "median"),
                                              obs_q90=("ddG", lambda x: x.quantile(.9)))


def line_fit(d, pred):
    """Slope and intercept of observed on predicted: 1 and 0 for a prediction that is right on average at every level."""
    slope, intercept = np.polyfit(d[pred], d["ddG"], 1)
    return {"slope": slope, "intercept": intercept, "sd_pred": d[pred].std(), "sd_obs": d["ddG"].std(),
            "mean_resid": (d[pred] - d["ddG"]).mean()}


def complex_bootstrap(d, fn, n=N_BOOT, seed=0):
    """Resample whole complexes with replacement (as in src.eval.metrics.bootstrap) and collect fn(resample)."""
    return [fn(r) for r in resample_complexes(d, n, seed)]


cal, summ = [], []
for name in MODELS:
    edges = np.quantile(base[name], np.linspace(0, 1, N_BINS + 1))
    t = by_pred_bin(base, name, edges)
    reps = pd.concat([by_pred_bin(r, name, edges)["mean_obs"] for r in complex_bootstrap(base, lambda r: r)], axis=1)
    t["obs_lo"], t["obs_hi"] = reps.quantile(.025, axis=1), reps.quantile(.975, axis=1)
    cal.append(t.assign(model=name, pred_from=edges[:-1], pred_to=edges[1:]).reset_index())
    for stratum, g in [("all", base)] + [(f"{s} side", g) for s, g in base.groupby("side")] + \
                      [("seen antigen" if s else "unseen antigen", g) for s, g in base.groupby("antigen_seen")]:
        fit = line_fit(g, name)
        slopes = [f["slope"] for f in complex_bootstrap(g, lambda r: line_fit(r, name))]
        summ.append({"model": name, "stratum": stratum, "rows": len(g), **fit,
                     "slope_lo": np.quantile(slopes, .025), "slope_hi": np.quantile(slopes, .975)})
cal, summ = pd.concat(cal, ignore_index=True), pd.DataFrame(summ)
cal.to_csv(OUT / "calibration_by_pred_bin.csv", index=False)
summ.to_csv(OUT / "calibration_summary.csv", index=False)
print(cal.round(2).to_string(index=False))
print(summ.round(2).to_string(index=False))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
lim = [-0.6, 3.6]
for name, colour in zip(MODELS, ["C0", "C1"]):
    t = cal[cal.model == name]
    ax[0].errorbar(t["mean_pred"], t["mean_obs"], yerr=[t["mean_obs"] - t["obs_lo"], t["obs_hi"] - t["mean_obs"]], fmt="o-",
                   color=colour, capsize=3, label=name)
t = cal[cal.model == MAIN]
ax[0].fill_between(t["mean_pred"], t["obs_q10"], t["obs_q90"], color="C0", alpha=.12, label=f"{MAIN}: 10-90% of observed")
ax[0].plot(lim, lim, "k--", lw=1)
ax[0].set(xlabel="mean predicted ddG in bin (kcal/mol)", ylabel="mean observed ddG in bin", xlim=lim, ylim=[-1.2, 5],
          title="Calibration by decile of the prediction\n(bars: 95% CI over complexes)")
ax[0].legend(fontsize=8, loc="upper left")
edges_obs = [-9, -0.5, 0.5, 1, 2, 3, 9]
t = base.assign(bin=pd.cut(base["ddG"], edges_obs)).groupby("bin", observed=True).agg(obs=("ddG", "mean"), pred=(MAIN, "mean"),
                                                                                    rows=("ddG", "size"))
x = np.arange(len(t))
ax[1].bar(x - .2, t["obs"], .4, label="mean observed")
ax[1].bar(x + .2, t["pred"], .4, label="mean predicted")
ax[1].set_xticks(x, [f"{i}\nn={n}" for i, n in zip(t.index.astype(str), t["rows"])], fontsize=8)
ax[1].set(xlabel="bin of the observed ddG", ylabel="kcal/mol", title=f"{MAIN}: the other direction, by bin of the observed value")
ax[1].legend(fontsize=8)
for side, colour in [("antibody", "C2"), ("antigen", "C3")]:
    g = base[base.side == side]
    t = by_pred_bin(g, MAIN, np.quantile(g[MAIN], np.linspace(0, 1, 6)))
    s = summ[(summ.model == MAIN) & (summ.stratum == f"{side} side")].iloc[0]
    ax[2].plot(t["mean_pred"], t["mean_obs"], "o-", color=colour, label=f"{side} side (n={len(g)}), slope {s['slope']:.2f}")
ax[2].plot(lim, lim, "k--", lw=1)
ax[2].set(xlabel="mean predicted ddG in bin", ylabel="mean observed ddG in bin", xlim=lim, ylim=[-1.2, 5],
          title=f"{MAIN}: calibration by side (quintiles)")
ax[2].legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "calibration.png", dpi=130)
plt.close(fig)


# ---------------------------------------------------------------- structural case studies
@lru_cache(maxsize=None)
def _original_numbering(pdb_id):
    """(chain, file residue number) -> original residue number of the SKEMPI mapping."""
    return {(c, v[1]): k for (c, k), v in read_mapping(pdb_id).items()}


def label(res):
    """Residue label in the original PDB numbering (the provided files are renumbered), e.g. D32.H"""
    chain = res.get_parent().id
    inv = _original_numbering(res.get_parent().get_parent().get_parent().id)
    return f"{AA3.get(res.get_resname(), 'X')}{inv.get((chain, res.id[1]), res.id[1])}.{chain}"


def ring_centre(res):
    names = RINGS.get(res.get_resname())
    return np.mean([res[n].coord for n in names], axis=0) if names and all(n in res for n in names) else None


def case_study(r):
    """Contacts of the mutated residue with the partner in the wild-type structure, and what the substitution removes."""
    model, res, partner = locate_site(r["complex"], r["mut_chain"], int(r["file_resnum"]))
    ns = NeighborSearch([a for c in partner if c in model for x in model[c] if x.id[0] == " " for a in heavy(x)])
    # atoms that the substitution removes: the whole side chain for Gly, beyond CB otherwise (exact for Ala, approximate for the rest)
    lost = {a.get_id() for a in heavy(res) if a.get_id() not in BACKBONE and (r["mut_aa"] == "G" or a.get_id() != "CB")}
    pairs = []
    for a in heavy(res):
        for p in ns.search(a.coord, 4.5):
            d = float(np.linalg.norm(a.coord - p.coord))
            polar = a.element in POLAR and p.element in POLAR and d <= 3.5
            salt = polar and (res.get_resname(), a.get_id()) in CHARGED_ATOMS and \
                (p.get_parent().get_resname(), p.get_id()) in CHARGED_ATOMS
            pairs.append({"partner": p.get_parent(), "atom": a.get_id(), "partner_atom": p.get_id(), "dist": d,
                          "polar": polar, "salt": salt, "lost": a.get_id() in lost, "a": a, "p": p})
    contacts = []
    for pres in sorted({x["partner"] for x in pairs}, key=lambda x: (x.get_parent().id, x.id[1])):
        pp = [x for x in pairs if x["partner"] is pres]
        kinds = []
        if any(x["salt"] for x in pp):
            kinds.append("salt bridge")
        elif any(x["polar"] for x in pp):
            kinds.append("H-bond")
        rc, pc = ring_centre(res), ring_centre(pres)
        if rc is not None and pc is not None and np.linalg.norm(rc - pc) <= 6.5:
            kinds.append("aromatic stacking")
        for ring, other in ((rc, pres), (pc, res)):
            cat = [other[n].coord for rn, n in CATIONS if other.get_resname() == rn and n in other]
            if ring is not None and cat and np.linalg.norm(ring - cat[0]) <= 6.0:
                kinds.append("cation-pi")
        contacts.append({**r[KEY].to_dict(), "partner_residue": label(pres), "atom_pairs_4.5": len(pp),
                         "pairs_from_removed_atoms": sum(x["lost"] for x in pp), "min_dist": min(x["dist"] for x in pp),
                         "polar_pairs_3.5": "; ".join(f"{x['atom']}-{x['partner_atom']} {x['dist']:.1f}" for x in pp if x["polar"]),
                         "interaction": ", ".join(kinds) or "van der Waals"})
    partner_atoms = {id(x["p"]) for x in pairs}
    summary = {"residue": label(res), "side_chain_atoms": len([a for a in heavy(res) if a.get_id() not in BACKBONE]),
               "removed_atoms": len(lost), "partner_residues_4.5": len(contacts), "partner_atoms_4.5": len(partner_atoms),
               "partner_atoms_4.5_of_removed": len({id(x["p"]) for x in pairs if x["lost"]}),
               "polar_pairs": sum(x["polar"] for x in pairs), "polar_pairs_of_removed": sum(x["polar"] and x["lost"] for x in pairs),
               "salt_bridge_pairs": sum(x["salt"] for x in pairs),
               "interactions": "; ".join(f"{c['partner_residue']} {c['interaction']}" for c in contacts
                                         if c["interaction"] != "van der Waals")}
    return res, pairs, contacts, summary


def draw(ax, res, pairs, title):
    """Mutated residue and the partner residues within 4.5 A, projected on the plane that best contains them."""
    shown = [res] + sorted({x["partner"] for x in pairs}, key=lambda x: (x.get_parent().id, x.id[1]))
    xyz = np.array([a.coord for x in shown for a in heavy(x)])
    centre = xyz.mean(axis=0)
    axes = np.linalg.svd(xyz - centre, full_matrices=False)[2][:2]

    def proj(c):
        return (np.asarray(c) - centre) @ axes.T
    for x in shown:
        mine = x is res
        at = heavy(x)
        for i, a in enumerate(at):
            for b in at[i + 1:]:
                if np.linalg.norm(a.coord - b.coord) < 1.95:
                    p, q = proj(a.coord), proj(b.coord)
                    ax.plot([p[0], q[0]], [p[1], q[1]], color="darkorange" if mine else "0.6", lw=3.5 if mine else 1.6,
                            solid_capstyle="round", zorder=3 if mine else 1)
        for a in at:
            if a.element in ("N", "O", "S"):
                ax.scatter(*proj(a.coord), s=38 if mine else 20, color={"N": "royalblue", "O": "crimson", "S": "gold"}[a.element],
                           zorder=4, edgecolor="k", linewidth=.3)
        far = max(at, key=lambda a: np.linalg.norm(proj(a.coord)))
        ax.annotate(label(x), proj(far.coord), fontsize=8 if mine else 7, fontweight="bold" if mine else "normal",
                    color="saddlebrown" if mine else "0.2", xytext=(3, 3), textcoords="offset points", zorder=5)
    for x in pairs:
        if x["polar"]:
            p, q = proj(x["a"].coord), proj(x["p"].coord)
            ax.plot([p[0], q[0]], [p[1], q[1]], ls=":", lw=1.6, color="magenta" if x["salt"] else "green", zorder=2)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def pymol_script(r, res, pairs):
    """A PyMOL script that opens the site: mutated residue in orange, partner residues within 4.5 A as sticks, polar contacts dashed."""
    ab, ag = antibody_chains(r["complex"])
    site = f"chain {r['mut_chain']} and resi {int(r['file_resnum'])}"
    lines = [f"# {r['complex']} {r['Mutation(s)_cleaned']}: observed ddG {r['ddG']:.2f}, {MAIN} predicted {r[MAIN]:.2f} kcal/mol",
             "# residue numbers are those of the SKEMPI-renumbered file; run from this folder: pymol <this file>",
             f"load ../../../data/SKEMPI2_PDBs/PDBs/{r['complex'][:4]}.pdb, cplx", "hide everything", "bg_color white",
             f"select antibody, chain {'+'.join(ab)}", f"select antigen, chain {'+'.join(ag)}",
             "show cartoon, antibody or antigen", "set cartoon_transparency, 0.6", "color grey80, antibody", "color lightblue, antigen",
             f"select site, {site}", f"select partner_near, byres ((chain {'+'.join(ag if r['mut_chain'] in ab else ab)}) within 4.5 of site)",
             "show sticks, site or partner_near", "util.cbao site", "util.cbag partner_near",
             "distance polar, site, partner_near, 3.5, mode=2", "label site and name CA, '%s%s' % (resn, resi)",
             "label partner_near and name CA, '%s%s.%s' % (resn, resi, chain)", "zoom site or partner_near, 4", "deselect"]
    (OUT / "pymol" / f"{r['complex']}_{r['Mutation(s)_cleaned']}.pml").write_text("\n".join(lines) + "\n")


d = base.assign(pred=base[MAIN])
d["resid"] = d["pred"] - d["ddG"]
d["pred_pct"] = d["pred"].rank(pct=True)
d["pred_pct_in_complex"] = d.groupby("complex")["pred"].rank(pct=True)
d["rows_in_complex"] = d.groupby("complex")["pred"].transform("size")
feat = d[STRUCT + SUBST].astype(float)
z = ((feat - feat.mean()) / feat.std()).values
worst = d.reindex(d["resid"].abs().sort_values(ascending=False).index).head(N_CASES)

cases, contact_rows, site_rows = [], [], []
fig, axs = plt.subplots(3, 4, figsize=(19, 14))
for ax, (i, r) in zip(axs.ravel(), worst.iterrows()):
    res, pairs, contacts, summary = case_study(r)
    contact_rows += contacts
    # rows of other complexes that look the same to the model: nearest in the standardised geometry + substitution features
    other = np.where((d["complex"] != r["complex"]).values)[0]
    near = d.iloc[other[np.argsort(np.linalg.norm(z[other] - z[d.index.get_loc(i)], axis=1))[:N_ANALOGUES]]]["ddG"]
    site = d[(d["complex"] == r["complex"]) & (d["mut_chain"] == r["mut_chain"]) & (d["file_resnum"] == r["file_resnum"])]
    site_rows += site[KEY + ["ddG", "pred"]].assign(case=f"{r['complex']} {r['Mutation(s)_cleaned']}").to_dict("records")
    cases.append({**r[KEY + ["antigen_group", "side", "iMutation_Location(s)", "Method", "n_repeats"]].to_dict(),
                  "kd_wt": r["Affinity_wt_parsed"], "kd_mut": r["Affinity_mut_parsed"], "ddG": r["ddG"], "pred": r["pred"],
                  "pred_pct": r["pred_pct"], "pred_pct_in_complex": r["pred_pct_in_complex"], "rows_in_complex": r["rows_in_complex"],
                  **summary, "rsa_complex": r["rsa_complex"], "dsasa": r["dsasa"], "bfactor": r["bfactor"],
                  "analogues_ddG_median": near.median(), "analogues_ddG_q90": near.quantile(.9), "analogues_ddG_max": near.max(),
                  "analogues_above_3": int((near > 3).sum()),
                  "same_site_rows": len(site), "same_site_ddG": "; ".join(f"{m[-1]} {v:.1f}" for m, v in
                                                                          zip(site["Mutation(s)_cleaned"], site["ddG"])),
                  "notes": r["Notes"]})
    draw(ax, res, pairs, f"{r['complex'][:4]} {summary['residue']} -> {r['mut_aa']} ({r['side']} side)\n"
                         f"observed {r['ddG']:+.1f}, predicted {r['pred']:+.1f}\nburied {1 - r['rsa_complex']:.0%}, "
                         f"{summary['partner_atoms_4.5']} partner atoms, {summary['polar_pairs']} polar pairs")
    pymol_script(r, res, pairs)
fig.suptitle(f"{MAIN}: the {N_CASES} largest errors. Mutated residue in orange, partner residues within 4.5 A in grey; "
             "dotted green = polar contact <= 3.5 A, dotted magenta = salt bridge", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, .97])
fig.savefig(OUT / "case_studies.png", dpi=120)
plt.close(fig)

cases, contact_rows = pd.DataFrame(cases), pd.DataFrame(contact_rows)
cases.to_csv(OUT / "case_studies.csv", index=False)
contact_rows.to_csv(OUT / "case_contacts.csv", index=False)
pd.DataFrame(site_rows).to_csv(OUT / "case_same_site.csv", index=False)
pd.set_option("display.width", 250)
print(cases.drop(columns=["notes", "interactions", "same_site_ddG"]).round(2).to_string(index=False))
print(cases[KEY + ["interactions", "same_site_ddG"]].to_string(index=False))
print(contact_rows.round(2).to_string(index=False))

# how unusual are the worst errors' structural features in the whole set? (percentile of each case within all 668 rows)
pct = d[STRUCT].rank(pct=True).loc[worst.index].assign(case=(worst["complex"].str[:4] + " " + worst["Mutation(s)_cleaned"]).values)
print(pct.set_index("case").round(2).to_string())
# and the reverse question: of all rows that look like a hotspot to the features, how many are one?
hot = d[(d["rsa_complex"] < .1) & (d["partner_atoms_4.5"] >= 8)]
print(f"buried (<10% accessible) with >= 8 partner atoms: {len(hot)} rows, observed ddG quartiles "
      f"{hot['ddG'].quantile([.25, .5, .75]).round(2).tolist()}, share above 3 kcal/mol {(hot['ddG'] > 3).mean():.0%}, "
      f"mean predicted {hot['pred'].mean():.2f}")
