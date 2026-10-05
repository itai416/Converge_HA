# Predicting antibody–antigen ΔΔG from sequence and structure (SKEMPI 2.0, AB/AG subset)

Converge Bio ML Researcher home assignment. **Work in progress:** stages 1–3 (data, EDA, split) are done,
stage 4 (features) is done for ESM-2 35M and ESM-IF1, and the ablation ladder is complete up to rung E (§6.2–6.3). Error analysis is in §6.4. The full plan is in [PLAN.md](PLAN.md).

**Task.** Predict ΔΔG = RT·ln(Kd_mut / Kd_wt) in kcal/mol (positive = the mutation weakens binding) for a mutation in
an antibody–antigen complex, from the protein sequence and the 3D structure of the wild-type complex.

**Framing.** We treat this as regression on ΔΔG. Our use case is **lead optimisation against a known antigen**: for a given antibody,
rank candidate mutations. We do not claim generalisation to new targets. This use case drives the split and the primary metric.

---

## 1. Data

| Step | Rows | Complexes |
|---|---|---|
| SKEMPI 2.0 rows with `Hold_out_type` containing AB/AG | 1,211 | 55 |
| single-point, uncensored, valid affinity | 756 | 47 |
| **v1 set** after averaging repeated measurements of the same (complex, mutation) | **668** | 46 |

- **Scope.** v1 covers single-point, uncensored mutations only. Multi-point rows (381) and censored single-point rows (74; for 24 unique rows a numeric lower bound on ΔΔG is known, see §6.7)
  are kept for later extensions (Tobit-style loss, additivity models).
- **Residue mapping.** Residues are located through SKEMPI's `.mapping` files, which handle PDB numbering and insertion codes. All 668 wild-type residues
  match the structure.
- **Antigen groups.** Antigens are grouped (e.g. lysozyme, gp120, integrin α-1). Anti-idiotype complexes (an antibody against another
  antibody, e.g. 1DVF) are kept as their own antigen class.

Code: [src/data/](src/data/), [scripts/01_clean.py](scripts/01_clean.py).

## 2. What the EDA told us, and what we did about it

Figures are in [results/eda/](results/eda/).

| Finding | Figure | Consequence |
|---|---|---|
| ΔΔG is right-skewed (skew 1.17, median 0.65), with a spike at 0 and a real tail up to +7.4. Only ~7% of rows are improving (< −0.5). | [01](results/eda/01_ddg_hist.png) | Robust loss (Huber) compared against MSE. Detecting improving mutations is hard and is reported separately. |
| Very uneven complexes: 3HFM, 1MHP and 1JRH dominate. Only 20 complexes have ≥10 v1 rows. | [02](results/eda/02_rows_per_complex.png), [03](results/eda/03_ddg_per_complex.png) | Grouped CV by complex. Per-complex metric averaged over complexes, not rows. Tested 1/√n complex weighting. |
| Repeated measurements agree to ~0.1–0.3 kcal/mol. | [05](results/eda/05_repeats_mean_std.png) | Experimental noise is not the bottleneck: the noise ceiling on explained variance is about 0.97. |
| Knowing the **complex** explains 26% of ΔΔG variance, location class 12%, wild-type amino acid 10%, the to-Ala flag 1%. | [06](results/eda/06_variance_explained.png) | A large part of ΔΔG is a per-complex offset, so pooled metrics can look good without ranking mutations well (see §4). |
| Interface core and support residues have large, widely spread ΔΔG. Non-interface surface residues are near 0. | [07](results/eda/07_location_class.png) | The structural position of the mutation matters. This motivated the geometry features and models B/C. |
| Mutating K, Y, W, H or D is most destabilising. To-Ala vs other substitutions barely differ. | [08](results/eda/08_substitution.png) | Wild-type identity is informative. Alanine-scan bias is not a big concern. |
| Simple substitution descriptors are weak alone (Δvolume ρ = −0.20, Δhydropathy 0.16, Δcharge 0.06). | [09](results/eda/09_descriptors.png) | The substitution alone is not enough. It has to be combined with the context. |

## 3. Evaluation design: what we check, and why

### 3.1 Split: grouped by complex, stratified by antigen
- **Why grouped.** Mutations of the same complex are strongly correlated (the complex alone explains 26% of the variance). A random split would let the
  model memorise each complex's offset and overstate performance. So **no complex appears in both train and test**.
- **Why stratified by antigen.** It keeps the antigen mix similar across folds. Lysozyme (13 complexes) is spread over all folds. This matches the
  "known antigen, new antibody" use case. A test complex whose antigen also appears in training is tagged *seen antigen*. Because lysozyme, gp120, integrin and others
  have several complexes, 36 of the 46 complexes (504 rows) are *seen* and 10 (164 rows) are *unseen*. We report the two groups separately (§6.4).
- **Repeated CV.** One partition of 46 complexes into 5 folds is a single draw, and results move noticeably with it. Every experiment is
  therefore run on **5 different fold assignments**, chosen as the 5 most size-balanced seeds (their mutual adjusted Rand index is about 0, so
  they really are different). LightGBM also runs with **3 seeds**. Repeat 0 is the saved split in `data/processed/folds.csv`.
- **Nested tuning.** All hyperparameters (penalty strength, number of PCA components, tree size, number of boosting rounds) are chosen by grouped
  inner CV inside each training fold, scored by pooled Spearman. **The test fold is never used for any choice.**

Code: [src/data/splits.py](src/data/splits.py), [src/models/cv.py](src/models/cv.py), [src/models/run.py](src/models/run.py).

### 3.2 Three metrics, each answering a different question

| Metric | Question | Why we keep it |
|---|---|---|
| **Per-complex Spearman** (primary): mean over the 20 complexes with ≥10 test rows | "For this antibody, are its mutations ranked correctly?" | Matches the lead-optimisation use case, and cannot be inflated by guessing a complex's overall ΔΔG level. |
| **Pooled Pearson** over all test rows | "How do we compare with published SKEMPI numbers?" | It is the standard benchmark metric. It is always more optimistic than the per-complex number, because it also rewards getting between-complex differences right. |
| **RMSE** (kcal/mol) | "Are the predicted values the right size?" | The two correlations ignore scale. Reference points: predicting the training mean gives about 1.56; experimental repeat noise is about 0.2. |

We deliberately dropped MAE (redundant with RMSE), pooled Spearman (redundant with pooled Pearson) and classification scores. Improving-mutation
AUPRC is the most use-case-relevant score, but there are only about 49 improving rows, so it is noisy. It will come back in the error analysis.

**How uncertainty is reported.**
- **mean ± std over the 5 repeats × seeds:** how much a result depends on the fold assignment and seed.
- **95% bootstrap CI over complexes** on the primary split: how much it depends on which complexes are in the dataset. About ±0.13 for per-complex Spearman.
- **Paired differences on identical folds** to decide whether model X beats model Y. Overlapping CIs alone don't answer that.

**Pitfall worth knowing.** The training-mean baseline gets pooled Pearson **−0.22, not 0**. The baseline is constant within a fold but changes
between folds: when the test fold has high ΔΔG, the training folds have low ΔΔG, so its prediction moves against the truth. This is one more reason
the primary metric is per-complex.

### 3.3 Baselines: what every model must beat
1. **Training mean.** The floor.
2. **Zero-shot ESM-2 log-likelihood ratio**, log p(mut) − log p(wt), mapped to kcal/mol with a 2-parameter line. **Purpose: if a trained model on ESM-2 features can't beat
   ESM-2's own zero-shot score, the regressor adds nothing.**
3. **Location class + wild-type/mutant amino acid** (one-hot features, ridge regression). This is the "cheap knowledge" baseline. It uses SKEMPI's location label, which is
   itself a crude structural feature.

## 4. Model choice

### 4.1 Which pretrained models, and why
The assignment lists several models. The deciding facts for our data:
- every complex has an experimental structure
- 58% of mutations are on the antibody and 42% on the antigen
- only 668 labelled rows

| Model | Kind | Covers antibody / antigen side | Sees the binding partner | Verdict |
|---|---|---|---|---|
| **ESM-2** (8M/35M/650M) | protein language model | ✓ / ✓ | ✗ | **Sequence modality.** Cheap, covers both sides. Scores fit to the protein, not to the interface. |
| ESMC 300M | protein language model | ✓ / ✓ | ✗ | Alternative to ESM-2 650M. Needs a newer Python than ours, so Colab only. |
| AbLang2 | antibody language model | ✓ / ✗ | ✗ | Less germline bias on CDRs, but no feature for 42% of rows. Optional add-on. |
| ESMFold, IgFold | structure **prediction** (sequence → structure) | ✓ / ✓ (IgFold antibody only) | weak / ✗ | **Skipped.** We already have better (experimental) structures. A single point mutation barely changes the predicted backbone, and neither models the antibody–antigen pose well. |
| **ESM-IF1** | **inverse folding** (structure → amino-acid probabilities) | ✓ / ✓ | **✓** | **Structure modality (planned, model D).** Answers "does this residue fit this 3D environment?" given the partner chains. Scoring with and without the partner gives an interface-specific signal. |
| AntiFold | antibody-tuned inverse folding | ✓ / ✗ | ✓ | Optional antibody-side upgrade of ESM-IF1. |
| SaProt | structure-aware language model | ✓ / ✓ | ✗ (structure tokens per chain) | Possible extension. Fusing inside the model would also blur the sequence-vs-structure ablation that the assignment grades. |

**Why folding models don't help here.** Structure prediction answers "what shape will this sequence take?", and the crystal structure has
already answered that better. The question we need answered is "is this mutation acceptable at this position in 3D, next to this partner?"
That is the inverse problem, which inverse-folding models such as ESM-IF1 address directly.

### 4.2 Architecture: frozen encoders + a small regressor
```
mutated chain ──► ESM-2 (frozen) ──► site features ─┐
wild-type complex ──► geometry (and later ESM-IF1) ─┼──► regressor (ridge or LightGBM) ──► ΔΔG
```
- **Frozen encoders, no fine-tuning.** With 668 rows from 46 complexes, fine-tuning a language model would memorise the complexes. Encoders are run once and their outputs cached.
- **Only the mutated chain goes into ESM-2.** ESM-2 was trained on single chains. Interaction with the partner is the structure modality's job,
  which keeps the two modalities cleanly separable for the ablation.
- **Mutation representation**, all at the mutated position: the ESM-2 masked log-likelihood ratio (1 number), the wild-type residue embedding
  (context), and the mutant-minus-wild-type embedding (the change). ESM-2 35M gives 480 numbers per embedding, so 961 features in total.
- **The regressor predicts ΔΔG directly.** Two regressors are compared:
  - **Augmented ridge** (after Hsu et al. 2022, *Nat. Biotechnol.*): a single linear layer on the full standardised embeddings, plus the zero-shot
    score with a lighter penalty, so the model can lean on the pretrained prior and learn corrections from the labels. A linear probe on frozen
    embeddings is the most common small-data recipe in the protein-LM literature.
  - **LightGBM**: an additive ensemble of shallow trees (4–7 leaves, up to 300 rounds). It is used because it can learn **interactions** (e.g.
    "unfavourable substitution × at the interface"), which a linear model cannot.
  - No deep network: with about 530 training rows per fold, it would have more capacity than the data supports. A gated-fusion MLP stays an optional
    later comparison.

### 4.3 Why compression (PCA) is used only where it is needed
961 features vs about 530 training rows means more parameters than examples. To show what that does, we trained models on **960 columns of pure
random noise** with our real labels:

| Model | Train Spearman | Test Spearman |
|---|---|---|
| plain linear regression | 1.00 | 0.07 |
| ridge (tuned penalty) | 0.71 | 0.13 (chance on one fold) |
| LightGBM | 1.00 | 0.06 |

- **Ridge** protects itself through its penalty, so it uses the **full embeddings**. PCA (16/64 components) is offered to it as a tuned option. Inner CV chose
  the full embeddings in 3 of 5 folds.
- **LightGBM** splits on one column at a time, and with hundreds of noisy columns some split always looks good by chance. Its embeddings are
  compressed with PCA (8/16/32 components, tuned; PCA fit on the training fold only).
- **For fusion**, compressing the embedding block also stops ~1,000 embedding columns from outweighing ~10 geometry features just by count.

### 4.4 Loss and sample weighting
- **Huber loss (δ = 1 kcal/mol) as the default, MSE as the ablation.** The tail is real signal (noise is about 0.2), so δ is set high enough that typical errors
  are still squared and only errors above 1 kcal/mol are down-weighted.
- **Complex weighting, 1/√(rows in complex), compared with unweighted.** Three complexes provide about a third of the training rows. This mirrors cluster
  reweighting in protein ML (e.g. sequence reweighting in EVmutation). √ is a compromise between no weighting and 1/n, which would overweight tiny,
  noisy complexes. There is no standard weighting in the SKEMPI ΔΔG literature; most methods train unweighted.

## 5. The ablation ladder: why each rung exists

Our question was whether "where the mutation is" (its distance to the partner) is enough structural information, or whether richer 3D
information helps. Each rung adds one thing, all on the same folds:

| Rung | Features | Question it answers | Status |
|---|---|---|---|
| A | sequence only (ESM-2) | What does a protein language model alone give? | done |
| B | A + distance to partner | Is "where" enough? | done (§6.2) |
| C | A + all geometry features (burial, contacts, H-bonds, ...) | Do richer hand-crafted 3D features help? | done (§6.2) |
| geometry only | geometry, no sequence model | the "structure only" arm of the modality ablation | done (§6.2) |
| D | C + ESM-IF1 scores (with / without partner) | Does a learned structure model add anything beyond geometry? | done (§6.3); features: [08_esmif1.py](scripts/08_esmif1.py) / Colab notebook |
| E | D + ESM-IF1 embedding | Do high-dimensional structure features help or overfit? | done (§6.3) |

Geometry features ([src/features/geometry.py](src/features/geometry.py)) are computed on the wild-type complex. The partner is the antigen when
an antibody residue is mutated, and vice versa. Single-feature Spearman with ΔΔG: partner atoms within 4.5 Å 0.45, distance to partner −0.43,
relative accessibility in the complex −0.40, ΔSASA on binding 0.35, cross-interface H-bonds 0.34. All are stronger than the substitution descriptors.

## 6. Results

### 6.1 Model A (sequence only, ESM-2 35M)
5 fold assignments; LightGBM × 3 seeds; unweighted rows. Mean ± std over runs.

| Model | Loss | Per-complex Spearman ↑ | Pooled Pearson ↑ | RMSE ↓ |
|---|---|---|---|---|
| Training mean | – | – | −0.22 ± 0.05 | 1.56 |
| Zero-shot ESM-2 LLR (calibrated) | Huber | −0.02 ± 0.06 | −0.22 ± 0.03 | 1.58 |
| Location + substitution | Huber | **0.34 ± 0.01** | 0.28 ± 0.04 | 1.53 |
| A: augmented ridge | Huber | 0.17 ± 0.04 | 0.31 ± 0.05 | 1.49 |
| A: LightGBM + PCA | Huber | 0.28 ± 0.03 | 0.35 ± 0.04 | 1.45 |
| A: LightGBM + PCA | MSE | 0.25 ± 0.04 | **0.43 ± 0.04** | **1.40** |

Full grid: [results/model_A/summary.csv](results/model_A/summary.csv). Per run: `runs.csv`. Bootstrap CIs: `metrics_rep0.csv`.

**What we learned from A**
1. **Zero-shot ESM-2 35M has no binding signal** (≈ 0 in every repeat). It scores whether the mutant fits the protein, which is mostly about stability, not the
   interface. As a result, inner CV rarely chose to boost it, and the augmented-ridge idea does not pay off with this score.
2. **Sequence features improve pooled Pearson and RMSE, but not within-complex ranking.** Location + substitution beats the best A in
   **5 of 5** fold assignments (by 0.03–0.11). A single structural label ranks mutations better than all of the ESM-2 features, which is early
   evidence that 3D information has to carry the model.
3. **LightGBM beats ridge on ranking** (0.28 vs 0.17), so non-linear effects matter.
4. **Huber vs MSE is a real trade-off.** Huber ranks better within complexes in **10/10** paired comparisons (+0.025); MSE fits magnitudes better (pooled
   Pearson, RMSE) because it chases the large destabilising values. Since the primary metric is per-complex, **Huber is the default**.
5. **1/√n weighting gives no consistent benefit.** It helps ridge in 8/10 comparisons (+0.02), LightGBM in 4/10, and location + substitution in 2/10. **Unweighted is the default.**

### 6.2 Ablation A → B → C (sequence + structure)
Settings: Huber loss, unweighted, 5 fold assignments, LightGBM × 3 seeds. Tuned on the inner folds: LightGBM PCA size {8, 16, 32, 64},
leaves {4, 7}, rounds {25 … 500}; ridge α, boost and PCA (see §7 for why these grids). Mean ± std over runs; the last column is the
95% bootstrap CI over complexes on the primary split. (A LightGBM scores higher here than in §6.1 because of the wider grid.)

| Model | Per-complex Spearman ↑ | Pooled Pearson ↑ | RMSE ↓ | Per-complex CI (primary split) |
|---|---|---|---|---|
| Location + substitution | 0.34 ± 0.01 | 0.28 ± 0.04 | 1.53 | [0.22, 0.46] |
| A: augmented ridge | 0.17 ± 0.04 | 0.31 ± 0.05 | 1.49 | [0.03, 0.24] |
| A: LightGBM | 0.30 ± 0.04 | 0.40 ± 0.06 | 1.41 | [0.18, 0.45] |
| B: augmented ridge | 0.22 ± 0.05 | 0.31 ± 0.03 | 1.49 | [0.08, 0.27] |
| B: LightGBM | 0.34 ± 0.03 | 0.43 ± 0.02 | 1.39 | [0.19, 0.47] |
| geometry only: ridge | 0.44 ± 0.01 | 0.45 ± 0.01 | 1.38 | [0.29, 0.56] |
| geometry only: LightGBM | 0.42 ± 0.02 | 0.42 ± 0.02 | 1.41 | [0.34, 0.52] |
| **C: augmented ridge** | **0.46 ± 0.02** | **0.46 ± 0.03** | **1.38** | [0.36, 0.58] |
| C: LightGBM | 0.43 ± 0.02 | 0.46 ± 0.03 | 1.37 | [0.29, 0.55] |

Paired per-complex Spearman differences on identical folds ([paired.csv](results/ablation_ABC/paired.csv)):

| Comparison | Mean difference | Range over repeats | Repeats where the first is better |
|---|---|---|---|
| B − A (LightGBM) | +0.04 | +0.00 … +0.09 | 5/5 |
| C − B (LightGBM) | +0.09 | +0.07 … +0.12 | 5/5 |
| C − A (ridge) | +0.29 | +0.24 … +0.35 | 5/5 |
| C − geometry only (ridge) | +0.02 | +0.00 … +0.05 | 5/5 |
| C − geometry only (LightGBM) | +0.01 | −0.01 … +0.05 | 3/5 |
| C LightGBM − location + substitution | +0.08 | +0.06 … +0.11 | 5/5 |

**What we learned from the ablation**
1. **Distance alone (B) helps, but is not enough.** It adds +0.04 for LightGBM in every repeat, which only brings it level with the location + substitution
   baseline. For ridge it helps less (+0.05), because a linear model can't express "the substitution matters only near the interface".
2. **Richer geometry (C) is the big step.** Burial, contacts, ΔSASA and H-bonds add another +0.09 over distance alone (LightGBM) and lift ridge
   by +0.29 over A. C is the first model that clearly beats the location baseline (5/5 repeats). So the answer to "is distance enough?" is **no**:
   *how buried and how connected* the residue is matters beyond *how far* it is from the partner.
3. **Structure carries almost all of the signal. Sequence (ESM-2 35M) adds a small but consistent increment on top.** Geometry alone reaches 0.44.
   Adding the sequence features gives +0.02 per-complex (ridge, 5/5 repeats). For LightGBM the per-complex gain is +0.01 and not consistent (3/5),
   but pooled Pearson rises by +0.04 and RMSE falls by 0.04.
   This is the current justification for fusion: it helps, but modestly. The two planned upgrades (ESM-2 650M, ESM-IF1) target exactly this gap.
4. **With geometry, ridge catches up with LightGBM** (0.46 vs 0.43; ridge better in 4/5 repeats). The geometry features relate to ΔΔG roughly
   monotonically, so the non-linearity LightGBM needed for sequence-only A matters less. Once geometry is added, C needs far fewer boosting rounds
   than A (see §7).

### 6.3 Ablation C → D → E (ESM-IF1)
Same protocol as §6.2 ([scripts/09_ablation_DE.py](scripts/09_ablation_DE.py), ESM-2 35M; results in [results/ablation_DE_esm2_35M/](results/ablation_DE_esm2_35M/)). Mean ± std over runs.

| Model | Per-complex Spearman ↑ | Pooled Pearson ↑ | RMSE ↓ |
|---|---|---|---|
| zero-shot ESM-IF1 interface score (calibrated) | 0.09 | −0.05 | 1.59 |
| geometry only: ridge | 0.44 ± 0.01 | 0.45 ± 0.01 | 1.38 |
| structure (geometry + ESM-IF1 scores): ridge | 0.44 ± 0.01 | 0.43 ± 0.02 | 1.40 |
| C: augmented ridge | 0.46 ± 0.02 | 0.46 ± 0.03 | 1.38 |
| D: augmented ridge | 0.46 ± 0.02 | 0.44 ± 0.03 | 1.39 |
| E: augmented ridge | 0.47 ± 0.02 | 0.44 ± 0.02 | 1.39 |
| C: LightGBM | 0.43 ± 0.02 | 0.46 ± 0.03 | 1.37 |
| D: LightGBM | 0.44 ± 0.02 | 0.46 ± 0.02 | 1.37 |
| E: LightGBM | 0.43 ± 0.02 | 0.45 ± 0.02 | 1.38 |

Paired differences (per-complex Spearman, same folds): D − C is +0.001 (ridge, better in 2/5 repeats) and +0.012 (LightGBM, 4/5);
E − D is +0.005 (3/5) and −0.002 (3/5); structure scores added to geometry give +0.003 (4/5).

1. **ESM-IF1 adds essentially nothing beyond hand-crafted geometry.** The gains are within noise and not consistent across repeats. The same holds when
   ESM-IF1 is the only learned structure component.
2. **The zero-shot interface score is weak** (Spearman 0.09 per complex), far below any single geometry feature (about 0.4).
3. **The high-dimensional ESM-IF1 embedding (E) does not help**, in line with the small-data argument in §4.3.
4. So the conclusion of §6.2 stands: structure carries most of the signal, via cheap geometry. The ESM-2 650M comparison is still open (§9).

### 6.4 Stratified results and error analysis
[scripts/10_error_analysis.py](scripts/10_error_analysis.py) on the out-of-fold predictions of the primary split; tables and figure in
[results/error_analysis/](results/error_analysis/). C ridge: per-complex Spearman 0.48 (95% CI over complexes [0.35, 0.58]), pooled Pearson 0.49, RMSE 1.36.
D LightGBM: 0.44 [0.29, 0.55].

| Stratum (C ridge) | Rows | Complexes with ≥10 rows | Per-complex Spearman |
|---|---|---|---|
| seen antigen | 504 | 14 | 0.49 |
| unseen antigen | 164 | 6 | 0.47 |
| antibody-side mutations | 385 | 14 | 0.35 |
| antigen-side mutations | 283 | 10 | 0.60 |
| to Ala | 364 | 13 | 0.57 |
| other substitutions | 304 | 9 | 0.41 |
| location COR / SUP / SUR / RIM | 281 / 115 / 79 / 162 | 11 / 2 / 3 / 5 | 0.43 / 0.40 / 0.27 / 0.18 |

1. **Seen vs unseen antigen:** ridge is almost unchanged (0.49 vs 0.47), LightGBM drops (0.47 → 0.35). With 6 evaluable unseen complexes this is suggestive only; the linear
   geometry model looks more transferable than the tree model.
2. **Antibody-side mutations are ranked much worse than antigen-side ones** (0.35 vs 0.60), although they are 58% of the rows. This is the clearest target for improvement
   (CDR/framework annotation, an antibody-specific model such as AbLang2 or AntiFold). RIM mutations are the hardest location class.
3. **Strong regression toward the mean.** Mutations with ΔΔG > 2 (125 rows) have mean observed 3.45 and mean predicted 1.37. The 49 improving mutations (< −0.5) have mean
   observed −1.25 and mean predicted +0.60, so the model cannot find improving mutations.
4. **The worst errors are large hotspots or large improving mutations.** Seven of the 12 largest errors are in 3HFM (lysozyme, K96/K97 and Y50 / N31 hotspots, observed 5.7–7.3),
   and the two BoNT_A1 H1036A rows (observed 7.3–7.4) are the largest. None of these has repeated measurements, so label noise cannot be excluded.
   They have not yet been checked against the structures.
5. Per-complex Spearman is 0.4–0.8 for most complexes with ≥10 rows. The exceptions are 1MLC (−0.45, 11 rows) and 2BDN (0.19, 12 rows).

Not done yet: SHAP feature importance, structural case studies of the worst errors, and the calibration plot by bin.

### 6.5 Targeted follow-up experiments (chosen from §6.4)
Instead of running every remaining item in the plan, we picked four hypotheses that the results so far made most likely, and ran only those.
Everything uses ridge (the best and fastest model), the same 5 fold assignments and paired comparisons.

| # | Hypothesis | Result |
|---|---|---|
| 4 | Zero-shot ESM-2 650M has binding signal where 35M had none | **No.** Per-complex Spearman of the raw score is 0.07 (35M: 0.02); antibody side 0.13, antigen side 0.01. A full 650M ablation is therefore not justified. |
| 3 | The worst errors are label noise or mapping errors | **Mostly no.** See below. |
| 1 | Antibody-side mutations need side-specific effects (geometry × side interactions) | **No.** −0.009 per-complex Spearman vs geometry only, worse in 5/5 repeats. |
| 2 | Richer 3D context helps: partner/chain environment features (new, [environment.py](src/features/environment.py)) and wild-type/mutant identity descriptors | **No consistent gain** (table below). |

Hypothesis 2 on top of geometry-only ridge ([scripts/12_hypotheses.py](scripts/12_hypotheses.py), [results/hypotheses_geometry/](results/hypotheses_geometry/)):

| Features | Per-complex Spearman | Pooled Pearson | RMSE | Paired vs geometry (repeats better) |
|---|---|---|---|---|
| geometry (reference) | 0.436 | 0.447 | 1.383 | – |
| + side interactions (H1) | 0.427 | 0.438 | 1.392 | −0.009 (0/5) |
| + environment | 0.418 | 0.468 | 1.366 | −0.018 (1/5) |
| + identity | 0.446 | 0.369 | 1.487 | +0.010 (4/5, range −0.014 … +0.040) |
| + environment + identity | 0.440 | 0.406 | 1.448 | +0.004 (2/5) |
| everything + side interactions | 0.431 | 0.436 | 1.416 | −0.005 (0/5) |

- Environment features improve the between-complex metrics (pooled Pearson +0.02, RMSE −0.02) but not within-complex ranking, which is the primary metric. Identity descriptors
  help ranking slightly and inconsistently but hurt pooled Pearson and RMSE. We did not carry any variant to model C, since none cleared the bar of a consistent paired gain.
- **Conclusion:** the hand-crafted geometry set is close to what this data supports for ranking with a low-capacity model. Per-feature correlations of the new features
  with ΔΔG are up to 0.35 (partner aromatic residues), but they are largely redundant with burial and contact counts.

**Hypothesis 3, the worst errors** (checked in the raw SKEMPI rows and the geometry features, not yet in a structure viewer):
- 10 of the 12 are real hotspots that the model under-predicts, not noisy labels. The residues are fully buried (relative accessibility ≈ 0), in contact
  with 8–26 partner atoms, 2.5–3.0 Å from the partner, and the mutant affinity sits at the assay's detection limit (µM). The model sees "buried contact residue" but predicts only +1.2 to +1.8
  because the regression shrinks toward the mean (§6.4); the observed 6–7.4 values are also partly capped by the assay range.
- **2VIS IC89T (observed −4.9):** the SKEMPI note says the crystal structure is one of the mutants, taken as the wild type. The structure and label disagree, so this row is a data problem.
- **1JRH E45P (observed −3.8):** a mutation to proline at a mostly exposed site (relative accessibility 0.42); the backbone effect is not captured by any feature.
- No censoring leak: all rows with a ">" affinity are already excluded from v1.

### 6.6 Adding multi-point rows to the training set
First of the "more data" experiments ([scripts/13_multipoint_features.py](scripts/13_multipoint_features.py),
[scripts/14_multipoint_experiment.py](scripts/14_multipoint_experiment.py), [results/multipoint/](results/multipoint/)). We added the 272 uncensored multi-point rows
(30 complexes, 7 of them with no single-point rows; all 1,058 sites located and featurised) to the 668 single-point rows. Per-site features of model C are pooled over the
sites of a mutation set (sum, so that a linear model is additive over sites, or mean). Same folds and C ridge as before; test rows are scored separately.

| Training set | Pooling | Single-point test rows | Multi-point test rows | **All test rows (single + multi)** | Paired vs singles-only, all rows |
|---|---|---|---|---|---|
| singles only (= C) | sum | **0.460** | 0.053 (additivity baseline) | **0.332** | – |
| singles only | mean | 0.460 | 0.119 | 0.329 | −0.003 (3/5) |
| + multi | sum | 0.385 | 0.090 | 0.271 | −0.061 (0/5) |
| + multi | mean | 0.397 | 0.143 | 0.273 | −0.059 (0/5) |
| + multi, 1/√n complex weights | sum | 0.396 | 0.039 | 0.276 | −0.056 (1/5) |
| + multi, 1/√n complex weights | mean | 0.410 | 0.082 | 0.289 | −0.043 (0/5) |

(Per-complex Spearman, mean over 5 fold assignments; the last column is the paired difference on identical rows and folds.)

1. **Adding the multi-point rows hurt single-point ranking in 19 of 20 paired comparisons (4 variants × 5 repeats; −0.05 to −0.075 on average), and it also lowers the combined single + multi score** (−0.04 to −0.06, never better in more than 1 of 5 repeats). The multi-point rows themselves improve only with mean pooling (0.119 → 0.143, +0.09 paired vs the sum-pooled baseline, 5/5), which is not enough to offset the loss on single-point rows. Complex weighting reduces but does not remove the harm, so the imbalance
   (86 rows from one complex) is not the whole story.
2. **Multi-point ΔΔG is hard to predict from per-site features**: per-complex Spearman 0.05–0.14 and RMSE 2.0–2.7 kcal/mol (vs 1.38 for single-point).
   The sites interact (and many large sets are combinations of hotspot residues), and our per-site features cannot see that.
3. Mean pooling beats sum pooling on multi-point rows (0.119 vs 0.053 for the singles-only model), i.e. strict additivity over-predicts large sets.
4. **Conclusion:** naively mixing multi-point rows into the training set does not improve C. They stay a separate extension; untested remedies are an `is_multi` indicator, a small
   sample weight for multi rows, and a model of the interaction between sites.

### 6.7 Adding censored rows with a Tobit-style loss
[scripts/15_censored_features.py](scripts/15_censored_features.py), [scripts/16_censored_experiment.py](scripts/16_censored_experiment.py), [results/censored/](results/censored/).
**Which rows.** 74 single-point rows are censored: 16 with mutant Kd ">X", 14 with wild-type Kd "<X" and an exact mutant Kd (both give a lower bound ΔΔG ≥ L),
11 with both affinities "<" (the ratio of two limits has no direction, so no bound), and 33 non-binders without a number. We used the 30 rows with a lower bound,
24 after keeping the largest bound per (complex, mutation): 10 complexes, mean bound 3.6 kcal/mol. Complex 4I77 has 14 of them and no exact rows.
**Loss.** A censored row costs Huber(max(0, L − prediction)): a prediction at or above the bound is free (`cens_col` in `AugmentedRidge`). They are used in training only; the inner CV
scores exact rows only and the test metric uses the same exact rows as before, so every comparison is paired with C.

| Training set (C ridge) | Per-complex Spearman | Paired vs C | RMSE | Mean prediction for exact rows with ΔΔG > 2 (true mean 3.45) | Tail RMSE |
|---|---|---|---|---|---|
| exact rows only (C) | 0.460 | – | 1.379 | 1.34 | 2.54 |
| + censored, one-sided Huber (Tobit) | 0.455 | −0.005 (2/5) | 1.377 | 1.41 | 2.50 |
| + censored, bound used as exact label | 0.454 | −0.006 (1/5) | 1.377 | 1.41 | 2.51 |
| + censored, Tobit, 1/√n weights | 0.454 | −0.006 (1/5) | 1.408 | 1.19 | 2.66 |

1. **No gain.** Per-complex Spearman is unchanged within noise (−0.005), and the tail improves only slightly without weights (RMSE 2.54 → 2.50) and worsens with weights.
2. **The one-sided loss behaves like the naive label.** On held-out censored rows the model predicts below the bound 83–92% of the time (mean shortfall 2.2 kcal/mol), and for a prediction below the
   bound the one-sided loss equals the exact-label loss. This confirms the tail under-prediction, but 24 rows, concentrated in a few complexes, are too few to correct it.
3. **Conclusion:** censored rows are not worth adding to C in this form. The Tobit loss stays implemented for the later combination; the 33 non-binders (no number) could only be used with an assumed bound.

## 7. Training diagnostics
The full report, covering what each check measured, why, the results and the actions taken, is in
[reports/training_diagnostics.md](reports/training_diagnostics.md). The key points:

[scripts/diag_training_curves.py](scripts/diag_training_curves.py) refits model A on the primary split with the hyperparameters chosen in each fold.
- **Huber ridge**: all 350 fits of the tuning grid converge (L-BFGS reaches a relative gap of about 1e-12 within 10–617 iterations).
  [Curves](results/model_A/training_curves_ridge.png). MSE ridge is solved in closed form.
- **LightGBM**: training loss keeps falling, as boosting always does. Held-out loss levels off around round 130–160 with Huber and rises slightly
  by round 300 (+4%), which is mild overfitting. [Curves](results/model_A/training_curves_lgbm.png). Fix: from the ablation run on, the number of boosting rounds
  (100/200/300) is tuned on the inner folds. The outer-fold curves are used only as a diagnostic, never for selection.
- Differences between folds (which complexes are in the test fold) are much larger than the effect of training length.
- **Tuning-grid edges.** If inner CV keeps choosing the lowest or highest value of a grid, the optimum probably lies outside it. Every run now
  writes `grid_edges.csv` and warns when a hyperparameter (in a grid of ≥3 values) sits on an edge in ≥50% of folds. The first ablation run
  had two such cases: geometry-only ridge picked the lowest penalty (α = 1) in 19/25 folds, and geometry-only LightGBM the fewest rounds (100)
  in 61/75. We widened the grids (α down to 0.01 for the low-dimensional models; rounds from 25) and reran. Both edges are gone (32% and 4%), and
  **no result changed by more than 0.005**, so the geometry-only vs C comparison was already fair. Low penalties are not offered to the embedding models:
  with ~960 columns, α = 0.01 makes the Huber problem ill-conditioned (~1,900 L-BFGS iterations per fit), and those models choose α ≈ 100 anyway.
- **A was under-tuned on the high side.** Sequence-only A LightGBM picked the largest PCA size (32) in 81% of folds and the most rounds (300) in 61%.
  Adding PCA 64 and 500 rounds raised it from 0.28 to **0.30** per-complex Spearman (pooled Pearson 0.33 → 0.40), and the gaps B − A and C − A
  shrank accordingly (the table above uses the wider grid). A still sits on the top edges (64 components in 81%, 500 rounds in 71%). We
  stopped widening here: A is a reference rung, the last widening gained +0.02, and its held-out curve below is still falling only slowly.
  This is recorded as a limitation: a sequence-only model may need more capacity than we gave it.

### 7.1 Convergence diagnostics for the ablation models
[scripts/diag_convergence.py](scripts/diag_convergence.py) → [convergence.png](results/ablation_ABC/convergence.png). It answers three separate questions, all on
the primary split with the hyperparameters chosen in each fold:

1. **Did the optimiser reach the minimum?** (Huber ridge; the objective is convex, so there is a single minimum.) We refit **every configuration of every
   ridge tuning grid on every outer training set (510 fits)**. All 510 report success, after at most 645 L-BFGS iterations (limit 2,000).
   The gradient, which is zero at the minimum, shrank by at least 770× in every fit (median ~30,000×), and the objective settles to a relative change of ~1e-12.
2. **Where does more boosting stop helping?** (LightGBM; curves refit to 500 rounds, dots = the round inner CV chose in that fold.)
   - **A (sequence only):** held-out loss is still falling slowly at round 500, and inner CV chose 300–500 in every fold. A is a slow learner on
     the embeddings. It is not overfitting, it is under-fitted.
   - **C (+ geometry):** held-out loss flattens by about round 150 and stays flat. Inner CV's choices are spread over 100–500 (16% at an edge),
     which is what a flat optimum looks like: the number of rounds barely matters for C.
   - In both, the gap between folds (held-out loss about 0.36 to 0.80) is far larger than any effect of training length.
3. **Was each tuning grid wide enough?** `grid_edges.csv`: after the widenings above, the only hyperparameters still at an edge in ≥50% of folds are
   A LightGBM's PCA size and rounds (and B LightGBM's PCA size, 57%), discussed above. C and both geometry-only models are not at an edge.

### 7.2 Learning curve: would more data help?
[scripts/07_learning_curve.py](scripts/07_learning_curve.py) → [learning_curve.png](results/learning_curve/learning_curve.png). In every outer fold of all 5 fold
assignments, each model is trained on a random 25 / 50 / 75 / 100% of the training **complexes** (whole complexes; 3 draws per fraction; the same draws for
every model) and scored on the untouched test fold. Hyperparameters are fixed to each model's most frequently chosen setting in the ablation;
re-tuning at every size would cost ~40,000 extra fits, and a fixed penalty if anything makes the small-data points slightly pessimistic.

| Training complexes per fold | 9 | 18 | 28 | 37 (all) |
|---|---|---|---|---|
| C: augmented ridge | 0.40 | 0.44 | 0.45 | 0.46 |
| geometry only: ridge | 0.38 | 0.42 | 0.43 | 0.43 |
| A: LightGBM (sequence only) | 0.14 | 0.17 | 0.26 | 0.31 |

(held-out per-complex Spearman, mean over fold assignments and draws)

1. **The geometry models have plateaued: they are limited by their features, not by the amount of data.** Geometry-only gains just +0.003 from 75% to 100% of the
   complexes, and its training and held-out scores almost coincide (0.46 vs 0.43): it is not overfitting, there is simply no more signal for it to extract.
   **More complexes of the same kind will not improve it. Better structural features will** (ESM-IF1, models D/E).
2. **The sequence-only model is limited by data.** A rises steeply and has not levelled off (0.17 → 0.26 → 0.31), with a very large train/test gap
   (0.87 vs 0.31). At full size its RMSE (1.36) already matches C's. It is the component that would benefit most from more data, which motivates
   **pretraining on the non-antibody protein–protein interaction part of SKEMPI** (Stage 9) before fitting on AB/AG.
3. **C sits between the two:** it is nearly flat at the end (+0.003 from 75% to 100%), with a moderate train/test gap (0.56 vs 0.46). The sequence part of the
   fusion is the only branch still data-hungry, so the fusion gain (+0.02 today) could grow with more data.

(The A LightGBM full-data point, 0.31, is slightly above its tuned ablation score, 0.30, because here it uses one fixed configuration and one seed.)

## 8. Reproducing

Python 3.9. The pinned environment is in [requirements.txt](requirements.txt) (torch is the CPU build):

```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

This covers every step except computing the ESM-IF1 features, whose outputs are cached in `data/processed/esmif1.parquet`.
Recomputing them needs [requirements-esmif1.txt](requirements-esmif1.txt) (fair-esm, torch_geometric, biotite ≥ 1.0) on Python ≥ 3.10:
biotite 0.39, the last release for Python 3.9, does not import with NumPy 2. We ran that step on Colab.
`torch_scatter` is optional: [inverse_folding.py](src/features/inverse_folding.py) provides a tested pure-PyTorch replacement
for the one function fair-esm uses.

The AI prompt history is in [AI_PROMPTS.md](AI_PROMPTS.md).

| Step | Command | Runtime (Windows 11, 16-thread CPU, no GPU) |
|---|---|---|
| clean + dedup | `python scripts/01_clean.py` | seconds |
| folds | `python scripts/02_split.py` | seconds |
| EDA figures | `python scripts/eda_basic.py`, `python scripts/eda_features.py` | seconds |
| geometry features | `python scripts/03_geometry.py` | ~2 min |
| ESM-2 35M features | `python scripts/04_esm2.py` | ~4 min (650M: pass `facebook/esm2_t33_650M_UR50D`, run on Colab GPU) |
| model A (130 runs) | `python scripts/05_train_A.py 35M 5 3` | ~12 min on 15 workers |
| ablation A/B/C (85 runs) | `python scripts/06_ablation_ABC.py 35M 5 3` | ~22 min on 15 workers |
| training curves (model A, loss × weighting) | `python scripts/diag_training_curves.py` | ~2 min |
| convergence diagnostics (ablation) | `python scripts/diag_convergence.py` | ~3 min |
| learning curve (750 fits) | `python scripts/07_learning_curve.py` | ~2 min |
| ablation D/E (90 runs) | `python scripts/09_ablation_DE.py 35M 5 3` | ~13 h on 15 workers (much slower than A/B/C; cause not yet diagnosed) |
| environment features | `python scripts/11_environment_features.py` | ~10 s |
| targeted experiments (§6.5, 30 runs) | `python scripts/12_hypotheses.py geometry` | ~2 min |
| multi-point features (§6.6) | `python scripts/13_multipoint_features.py` | ~5 min |
| multi-point experiment (30 runs) | `python scripts/14_multipoint_experiment.py` | ~13 min on 15 workers |
| censored features (§6.7) | `python scripts/15_censored_features.py` | ~1 min |
| censored experiment (25 runs) | `python scripts/16_censored_experiment.py` | ~8 min on 15 workers |
| error analysis | `python scripts/10_error_analysis.py` | seconds |
| bundle for Colab | `python scripts/make_colab_bundle.py` → upload `colab_bundle.zip` to Drive | seconds |
| ESM-2 650M + ESM-IF1 features | [notebooks/02_colab_embeddings.ipynb](notebooks/02_colab_embeddings.ipynb) on a Colab T4 GPU | ~15–20 min including installs |
| ESM-IF1 features without a GPU | `python scripts/08_esmif1.py` (needs fair-esm, torch_geometric, biotite; checkpoints every 25 mutations, rerun to resume) | ~2.5 h on CPU |

Data: `data/skempi_v2.csv` and `data/SKEMPI2_PDBs/PDBs/` from the SKEMPI 2.0 website.

## 9. Limitations so far and next steps
- **ESM-2 35M only in the ablations.** The zero-shot 650M score is also uninformative (§6.5), so we did not rerun the ablation, but a trained model on 650M embeddings was not tested.
- **All features use the wild-type structure.** Conformational change on mutation is not modelled.
- **Only 20 complexes support the per-complex metric.** CIs are about ±0.13, so only differences that are consistent across paired repeats are claimed.
- **The antibody side** is assigned with a hand-written map from SKEMPI protein names, not by numbering the chains with ANARCI.
  CDR vs framework annotation is still missing.
- **The geometry models are feature-limited, the sequence model data-limited** (§7.2). This sets the priorities: better structural features first,
  then transfer from non-antibody SKEMPI data for the sequence branch.
- **Next:** SHAP on the final model and viewing the case studies in the structure; then the extensions. Transfer from non-antibody SKEMPI data is the only remaining idea
  aimed at the data-limited sequence branch; the other §6.5 ideas did not pay off.
  (multi-point mutations, censored rows).
