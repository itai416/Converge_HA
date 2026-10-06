# Features, in short

Every feature describes **one mutated residue in the wild-type complex**. No mutant structure is used.
Code: [src/features/](src/features/). Cached tables: `data/processed/`.

## 1. Geometry (structure), 10 features — [geometry.py](src/features/geometry.py)

Computed from atom coordinates in the experimental PDB file. "Partner" = the other side of the complex (antigen for an antibody residue, antibody for an antigen residue).

| Feature | What it is | Why it should matter | Spearman with ΔΔG |
|---|---|---|---|
| `on_antibody` | 1 if the residue is on the antibody, 0 if on the antigen | The two sides behave differently | – |
| `rsa_complex` | Relative solvent accessibility in the bound complex (0 = buried, ~1 = exposed) | Buried residues pack against the partner | −0.40 |
| `rsa_unbound` | Same, with the partner removed | Exposed alone but buried in the complex = interface residue | – |
| `dsasa` | Surface area (Å²) hidden by binding = unbound − bound | Direct size of the residue's interface contribution | 0.35 |
| `partner_atoms_4.5` | Partner heavy atoms within 4.5 Å of the residue | Number of contacts; the strongest single feature | 0.45 |
| `partner_atoms_8` | Same within 8 Å | Wider neighbourhood: interface centre vs edge | – |
| `min_dist_partner` | Shortest distance to any partner atom (capped at 12 Å) | Simple "how close to the partner" | −0.43 |
| `cross_hbonds` | Polar N/O atom pairs across the interface within 3.5 Å | Hydrogen bonds are specific, strong contacts | 0.34 |
| `cross_saltbridges` | Charged atom pairs across the interface within 3.5 Å | Electrostatic contacts | – |
| `bfactor` | Mean crystallographic B-factor of the residue | Flexibility proxy | – |

Known weaknesses:
- H-bonds are a distance count with no angle check, and they include backbone atoms.
- Salt bridges do not check that the two charges are opposite.
- `bfactor` depends on crystal resolution and is the least physical feature.
- All ten describe the wild-type site, so two substitutions at the same position get identical values.

## 2. Sequence (ESM-2 35M, frozen), 961 features — [sequence.py](src/features/sequence.py)

Only the **mutated chain** goes into the model; it never sees the partner.

| Feature | Count | What it is |
|---|---|---|
| `llr` | 1 | Zero-shot score: log p(mutant) − log p(wild-type) at the masked site. "Does this mutation fit the protein?" Mostly about stability, not binding |
| `wt_emb_*` | 480 | Embedding of the site in the wild-type chain: the residue's context |
| `diff_emb_*` | 480 | Site embedding in the mutant chain minus the wild-type one: the change caused by the mutation |

ESM-2 650M gives 1,280 + 1,280 + 1 = 2,561 features. It was extracted, but only its zero-shot score was evaluated.

## 3. Inverse folding (ESM-IF1), models D and E only — [inverse_folding.py](src/features/inverse_folding.py)

ESM-IF1 reads the backbone structure and predicts which residue fits each position.

| Feature | Count | What it is |
|---|---|---|
| `if_site_llr_{complex, unbound, interface}` | 3 | log p(mutant) − log p(wild-type) at the site, given the full complex, the own chain alone, and their difference |
| `if_seq_llr_{complex, unbound, interface}` | 3 | Same, using the likelihood of the whole mutant chain |
| `if_emb_complex_*` | ~1,024 | Encoder embedding of the site (model E only) |

`interface` = complex − unbound. It isolates what the partner contributes. It did not help beyond geometry (zero-shot 0.09, D and E ≈ C).

## 4. Baseline features (location + substitution)

One-hot encoding of SKEMPI's location class (COR, RIM, SUP, INT, SUR), the wild-type amino acid and the mutant amino acid. No structure is computed. It scores 0.34.

## 5. Computed but not in the main model

- **Substitution descriptors** (`d_volume`, `d_hydropathy`, `d_charge`, `blosum62`): used in the EDA; weak alone (Δvolume ρ = −0.20, Δhydropathy 0.16).
- **Environment features** ([environment.py](src/features/environment.py)): 20 columns such as nearby partner charge and hydrophobicity. Tested in README §6.5; did not help.

## 6. What each model receives

| Model | Features |
|---|---|
| A | ESM-2 (961) |
| B | A + `min_dist_partner` |
| C (best) | A + the 10 geometry features |
| Geometry only | the 10 geometry features |
| D | C + ESM-IF1 scores |
| E | D + ESM-IF1 embedding |
| Location + substitution | one-hot location and amino acids |
