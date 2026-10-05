# Predicting antibody–antigen ΔΔG from sequence and structure (SKEMPI 2.0, AB/AG subset)

Converge Bio ML Researcher home assignment. The stage-by-stage plan is in [PLAN.md](PLAN.md), the AI prompt history in [AI_PROMPTS.md](AI_PROMPTS.md).

**Task.** Predict ΔΔG = RT·ln(Kd_mut / Kd_wt) in kcal/mol (positive = the mutation weakens binding) for a mutation in
an antibody–antigen complex, from the protein sequence and the 3D structure of the wild-type complex.

**Framing.** Regression on ΔΔG. The use case is **lead optimisation against a known antigen**: for a given antibody, rank candidate
mutations. This drives the split and the primary metric. Complexes with an unseen antigen are reported separately (§6.4).

**Main results.**
- The fused model (C: ESM-2 embeddings + geometry features, ridge) reaches **per-complex Spearman 0.46 ± 0.02**, pooled Pearson 0.46, RMSE 1.38 kcal/mol,
  against 0.34 for the best baseline and 0.30 for the best sequence-only model.
- **Structure carries almost all of the signal.** Geometry alone gives 0.44; adding sequence gives a small, consistent +0.02.
- A learned structure model (ESM-IF1) adds nothing beyond hand-crafted geometry.
- The model compresses its predictions: it under-predicts hotspots and cannot find improving mutations. Antibody-side mutations are ranked much worse than antigen-side ones.
- Adding multi-point or censored rows to training did not help.

---

## 1. Data

| Step | Rows | Complexes |
|---|---|---|
| SKEMPI 2.0 rows with `Hold_out_type` containing AB/AG | 1,211 | 55 |
| single-point, uncensored, valid affinity | 756 | 46 |
| **v1 set** after averaging repeated measurements of the same (complex, mutation) | **668** | 46 |

- **Scope.** v1 is single-point, uncensored mutations. The 381 multi-point rows and 74 censored single-point rows were tested as extra training data in §6.6–6.7 and did not help.
- **Residue mapping.** Residues are located through SKEMPI's `.mapping` files (PDB numbering, insertion codes). All 668 wild-type residues match the structure.
- **Antigen groups.** Antigens are grouped (lysozyme, gp120, integrin α-1, ...). Anti-idiotype complexes (e.g. 1DVF) are their own class.

Code: [src/data/](src/data/), [scripts/01_clean.py](scripts/01_clean.py).

## 2. What the EDA told us, and what we did about it

Figures are in [results/eda/](results/eda/).

| Finding | Figure | Consequence |
|---|---|---|
| ΔΔG is right-skewed (median 0.65, tail to +7.4). Only ~7% of rows are improving (< −0.5). | [01](results/eda/01_ddg_hist.png) | Huber loss compared with MSE. Improving mutations are reported separately. |
| Complexes are very uneven: 3HFM, 1MHP and 1JRH dominate; only 20 have ≥10 rows. | [02](results/eda/02_rows_per_complex.png), [03](results/eda/03_ddg_per_complex.png) | Grouped CV by complex; metric averaged over complexes, not rows; 1/√n weighting tested. |
| Repeated measurements agree to ~0.1–0.3 kcal/mol. | [05](results/eda/05_repeats_mean_std.png) | Experimental noise is not the bottleneck (noise ceiling on explained variance ≈ 0.97). |
| The **complex** alone explains 26% of ΔΔG variance, location class 12%, wild-type amino acid 10%. | [06](results/eda/06_variance_explained.png) | Pooled metrics can look good without ranking mutations well, so the primary metric is per-complex. |
| Interface core/support residues have large, spread ΔΔG; non-interface surface is near 0. | [07](results/eda/07_location_class.png) | Structural position matters: this motivated the geometry features. |
| Substitution descriptors are weak alone (Δvolume ρ = −0.20, Δhydropathy 0.16). To-Ala vs other barely differ. | [08](results/eda/08_substitution.png), [09](results/eda/09_descriptors.png) | The substitution has to be combined with its context. |

## 3. Evaluation design

### 3.1 Split: grouped by complex, stratified by antigen
- **Grouped.** No complex appears in both train and test. A random split would let the model memorise each complex's offset (26% of the variance).
- **Stratified by antigen.** Keeps the antigen mix similar across folds, matching the "known antigen, new antibody" use case. A test complex whose
  antigen also appears in training is *seen* (36 complexes, 504 rows); the other 10 (164 rows) are *unseen*.
- **Repeated CV.** One 5-fold partition of 46 complexes is a single draw, so every experiment runs on **5 different fold assignments** (the 5 most
  size-balanced seeds; mutual adjusted Rand index ≈ 0). LightGBM also runs with 3 seeds.
- **Nested tuning.** All hyperparameters are chosen by grouped inner CV inside each training fold. The test fold is never used for any choice.

Code: [src/data/splits.py](src/data/splits.py), [src/models/cv.py](src/models/cv.py), [src/models/run.py](src/models/run.py).

### 3.2 Metrics

| Metric | Question | Note |
|---|---|---|
| **Per-complex Spearman** (primary), mean over the 20 complexes with ≥10 test rows | "For this antibody, are its mutations ranked correctly?" | Cannot be inflated by guessing a complex's overall ΔΔG level. |
| **Pooled Pearson** over all test rows | "How do we compare with published SKEMPI numbers?" | The standard benchmark metric; also rewards between-complex differences. |
| **RMSE** (kcal/mol) | "Are the predicted values the right size?" | Predicting the training mean gives 1.56; repeat noise is about 0.2. |

- **Uncertainty.** Mean ± std over repeats × seeds; a 95% bootstrap CI over complexes (about ±0.13 for per-complex Spearman); and **paired differences
  on identical folds** to decide whether one model beats another.
- **Pitfall.** The training-mean baseline gets pooled Pearson −0.22, not 0: its prediction is constant within a fold but moves against the truth
  between folds. One more reason the primary metric is per-complex.

### 3.3 Baselines
1. **Training mean.** The floor.
2. **Zero-shot ESM-2 log-likelihood ratio**, mapped to kcal/mol with a 2-parameter line. If a trained model can't beat it, the regressor adds nothing.
3. **Location class + wild-type/mutant amino acid** (one-hot, ridge). "Cheap knowledge"; SKEMPI's location label is itself a crude structural feature.

## 4. Model

### 4.1 Which pretrained models
Deciding facts: every complex has an experimental structure, 58% of mutations are on the antibody and 42% on the antigen, and there are only 668 labelled rows.

| Model | Kind | Covers antibody / antigen | Sees the partner | Verdict |
|---|---|---|---|---|
| **ESM-2** (35M; 650M zero-shot) | protein language model | ✓ / ✓ | ✗ | **Sequence modality.** Cheap, covers both sides. |
| **ESM-IF1** | inverse folding (structure → amino-acid probabilities) | ✓ / ✓ | ✓ | **Learned structure modality** (models D, E). Scored with and without the partner chains for an interface-specific signal. |
| ESMFold, IgFold | structure prediction | ✓ / ✓ (IgFold antibody only) | weak / ✗ | Skipped: we already have experimental structures, and a point mutation barely changes the predicted backbone. |
| AbLang2, AntiFold | antibody-specific | ✓ / ✗ | ✗ / ✓ | Not run: no features for the 42% antigen-side rows. A candidate next step for the antibody side (§9). |
| SaProt, ESMC 300M | structure-aware / newer language model | ✓ / ✓ | ✗ | Not run. SaProt fuses inside the encoder, which would blur the sequence-vs-structure ablation. |

Structure prediction answers "what shape will this sequence take?", which the crystal structure already answers. We need "is this mutation acceptable
at this position, next to this partner?", which is what geometry features and inverse folding address.

### 4.2 Representation and fusion
```
mutated chain ──► ESM-2 (frozen) ──► site features ──────────┐
wild-type complex ──► geometry features, ESM-IF1 scores ─────┼──► regressor (ridge or LightGBM) ──► ΔΔG
```
- **Mutation representation (sequence),** all at the mutated position: the masked log-likelihood ratio, the wild-type residue embedding (context) and the
  mutant-minus-wild-type embedding (the change): 961 features with ESM-2 35M. Only the mutated chain goes into ESM-2, which was trained on single chains.
- **Structure representation:** geometry of the mutated residue in the wild-type complex relative to the partner: distance, contacts, burial, ΔSASA on
  binding, cross-interface H-bonds ([src/features/geometry.py](src/features/geometry.py)). Models D/E add ESM-IF1 scores and embedding.
- **Fusion: concatenation of the per-modality feature blocks into one small regressor.** Why this and not a learned fusion network:
  - With 668 rows from 46 complexes, a gated or attention fusion has more capacity than the data supports. Frozen encoders are used for the same reason.
  - Separate blocks keep the modalities separable, so the ablation (sequence only, structure only, both) is clean.
  - Interaction between modalities is still available: LightGBM can learn "unfavourable substitution × at the interface". Ridge cannot, and comparing the two measures how much that matters.
  - The blocks are balanced explicitly: the embedding block is compressed with PCA for LightGBM, so ~1,000 embedding columns do not outweigh ~10 geometry features by count.
- **Regressors.**
  - **Augmented ridge** (after Hsu et al. 2022, *Nat. Biotechnol.*): a linear model on the standardised features, with a lighter penalty on the zero-shot score so the model can lean on the pretrained prior.
  - **LightGBM**: shallow trees (4–7 leaves), for non-linear effects and interactions.

### 4.3 Why PCA is used only where needed
961 features vs about 530 training rows. On **960 columns of pure noise** with the real labels, plain linear regression and LightGBM reach train Spearman 1.00
(test 0.07 and 0.06), and tuned ridge 0.71 (test 0.13). So:
- **Ridge** is protected by its penalty and uses the full embeddings; PCA is offered as a tuned option.
- **LightGBM** splits on one column at a time, and among hundreds of noisy columns some split always looks good by chance. Its embeddings are compressed with PCA (tuned; fit on the training fold only).

### 4.4 Loss and sample weighting
- **Huber (δ = 1 kcal/mol) vs MSE.** Huber ranks better within complexes in 10/10 paired comparisons (+0.025); MSE fits magnitudes better (pooled Pearson, RMSE). The primary metric is per-complex, so **Huber is the default**.
- **1/√(rows in complex) weighting vs unweighted.** Three complexes provide about a third of the rows. Weighting gives no consistent benefit (ridge 8/10, LightGBM 4/10, baseline 2/10), so **unweighted is the default**.

## 5. The ablation ladder

Each rung adds one thing, on the same folds:

| Rung | Features | Question |
|---|---|---|
| A | sequence only (ESM-2) | What does a protein language model alone give? |
| B | A + distance to partner | Is "where the mutation is" enough? |
| C | A + all geometry features | Do richer hand-crafted 3D features help? |
| geometry only | geometry, no sequence model | The "structure only" arm. |
| D | C + ESM-IF1 scores (with / without partner) | Does a learned structure model add anything beyond geometry? |
| E | D + ESM-IF1 embedding | Do high-dimensional structure features help or overfit? |

Single-feature Spearman with ΔΔG: partner atoms within 4.5 Å 0.45, distance to partner −0.43, relative accessibility −0.40, ΔSASA 0.35, cross-interface H-bonds 0.34.

## 6. Results

Protocol for all tables: ESM-2 35M, Huber loss, unweighted, 5 fold assignments, LightGBM × 3 seeds, nested tuning. Mean ± std over runs.

### 6.1 Baselines

| Model | Per-complex Spearman ↑ | Pooled Pearson ↑ | RMSE ↓ |
|---|---|---|---|
| Training mean | – | −0.22 ± 0.05 | 1.56 |
| Zero-shot ESM-2 35M (calibrated) | −0.02 ± 0.06 | −0.22 ± 0.03 | 1.58 |
| Location + substitution | 0.34 ± 0.01 | 0.28 ± 0.04 | 1.53 |

Zero-shot ESM-2 has no binding signal: it scores whether the mutant fits the protein, which is mostly about stability, not the interface.
Full model-A grid (loss × weighting): [results/model_A/summary.csv](results/model_A/summary.csv).

### 6.2 Ablation A → B → C

| Model | Per-complex Spearman ↑ | Pooled Pearson ↑ | RMSE ↓ |
|---|---|---|---|
| A: augmented ridge | 0.17 ± 0.04 | 0.31 ± 0.05 | 1.49 |
| A: LightGBM | 0.30 ± 0.04 | 0.40 ± 0.06 | 1.41 |
| B: augmented ridge | 0.22 ± 0.05 | 0.31 ± 0.03 | 1.49 |
| B: LightGBM | 0.34 ± 0.03 | 0.43 ± 0.02 | 1.39 |
| geometry only: ridge | 0.44 ± 0.01 | 0.45 ± 0.01 | 1.38 |
| geometry only: LightGBM | 0.42 ± 0.02 | 0.42 ± 0.02 | 1.41 |
| **C: augmented ridge** | **0.46 ± 0.02** | **0.46 ± 0.03** | 1.38 |
| C: LightGBM | 0.43 ± 0.02 | 0.46 ± 0.03 | 1.37 |

95% CI over complexes on the primary split, C ridge: [0.36, 0.58]. Paired per-complex differences on identical folds ([paired.csv](results/ablation_ABC/paired.csv)):

| Comparison | Mean difference | Range over repeats | Repeats better |
|---|---|---|---|
| B − A (LightGBM) | +0.04 | +0.00 … +0.09 | 5/5 |
| C − B (LightGBM) | +0.09 | +0.07 … +0.12 | 5/5 |
| C − A (ridge) | +0.29 | +0.24 … +0.35 | 5/5 |
| C − geometry only (ridge) | +0.02 | +0.00 … +0.05 | 5/5 |
| C − geometry only (LightGBM) | +0.01 | −0.01 … +0.05 | 3/5 |
| C LightGBM − location + substitution | +0.08 | +0.06 … +0.11 | 5/5 |

1. **Sequence alone ranks worse than one structural label.** The best A (0.30) is below location + substitution (0.34).
2. **Distance alone (B) helps but is not enough** (+0.04 LightGBM, +0.05 ridge): it only brings LightGBM level with the baseline.
3. **Richer geometry (C) is the big step** (+0.09 over B): how buried and how connected the residue is matters beyond how far it is from the partner.
4. **Fusion helps, modestly.** Sequence on top of geometry adds +0.02 per-complex for ridge (5/5). For LightGBM the gain is not consistent (3/5), though pooled Pearson rises by 0.04.
5. **With geometry, ridge catches up with LightGBM** (0.46 vs 0.43, better in 4/5): geometry relates to ΔΔG roughly monotonically, so the interactions LightGBM needed for A matter less.

### 6.3 Ablation C → D → E (ESM-IF1)
[results/ablation_DE_esm2_35M/](results/ablation_DE_esm2_35M/)

| Model | Per-complex Spearman ↑ | Pooled Pearson ↑ | RMSE ↓ |
|---|---|---|---|
| zero-shot ESM-IF1 interface score (calibrated) | 0.09 | −0.05 | 1.59 |
| geometry + ESM-IF1 scores: ridge | 0.44 ± 0.01 | 0.43 ± 0.02 | 1.40 |
| D: augmented ridge / LightGBM | 0.46 ± 0.02 / 0.44 ± 0.02 | 0.44 / 0.46 | 1.39 / 1.37 |
| E: augmented ridge / LightGBM | 0.47 ± 0.02 / 0.43 ± 0.02 | 0.44 / 0.45 | 1.39 / 1.38 |

**ESM-IF1 adds essentially nothing beyond geometry.** Paired: D − C is +0.001 (ridge, 2/5) and +0.012 (LightGBM, 4/5); E − D is +0.005 (3/5) and −0.002 (3/5).
Its zero-shot interface score (0.09) is far below any single geometry feature (about 0.4).

### 6.4 Error analysis
Out-of-fold predictions of the primary split, C ridge (per-complex Spearman 0.48, pooled Pearson 0.49, RMSE 1.36).
[scripts/10_error_analysis.py](scripts/10_error_analysis.py), [scripts/19_calibration_case_studies.py](scripts/19_calibration_case_studies.py), outputs in [results/error_analysis/](results/error_analysis/).

| Stratum | Rows | Complexes with ≥10 rows | Per-complex Spearman |
|---|---|---|---|
| seen / unseen antigen | 504 / 164 | 14 / 6 | 0.49 / 0.47 |
| antibody-side / antigen-side mutations | 385 / 283 | 14 / 10 | 0.35 / 0.60 |
| to Ala / other substitutions | 364 / 304 | 13 / 9 | 0.57 / 0.41 |
| location COR / SUP / SUR / RIM | 281 / 115 / 79 / 162 | 11 / 2 / 3 / 5 | 0.43 / 0.40 / 0.27 / 0.18 |

1. **Unseen antigens:** ridge is almost unchanged (0.49 vs 0.47), LightGBM drops (0.47 → 0.35). With 6 evaluable complexes this is suggestive only.
2. **Antibody-side mutations are ranked much worse** (0.35 vs 0.60), although they are 58% of the rows. This is the clearest target for improvement.
3. **Predictions are compressed toward the mean** ([calibration.png](results/error_analysis/calibration.png)). Their standard deviation is 0.59 against 1.54 observed. Mutations with ΔΔG > 2 (125 rows) have mean observed
   3.45 and mean predicted 1.37. The 49 improving mutations (< −0.5) have mean predicted +0.60, so the model cannot find improving mutations.
4. **The worst errors are real hotspots, not label noise.** 10 of the 12 largest errors are fully buried residues with 8–26 partner atoms in contact and identifiable interactions
   (H-bonds, cation–π, aromatic stacking; [case_studies.csv](results/error_analysis/case_studies.csv)), observed 6–7.4 with the mutant affinity at the assay's detection limit. Seven are in 3HFM.
   The other two: 2VIS IC89T (the crystal structure is a mutant taken as wild type, a data problem) and 1JRH E45P (proline at an exposed site; no feature captures the backbone effect).
5. Per-complex Spearman is 0.4–0.8 for most complexes with ≥10 rows. Exceptions: 1MLC (−0.45, 11 rows) and 2BDN (0.19, 12 rows).

### 6.5 Targeted follow-up experiments
Hypotheses suggested by §6.4, tested with ridge on the same folds ([scripts/12_hypotheses.py](scripts/12_hypotheses.py), [results/hypotheses_geometry/](results/hypotheses_geometry/)). Paired differences are vs geometry-only ridge (0.436).

| Hypothesis | Result |
|---|---|
| Zero-shot ESM-2 650M has binding signal where 35M had none | **No.** Per-complex Spearman 0.07 (antibody side 0.13, antigen side 0.01). A full 650M ablation was not run. |
| Antibody-side mutations need side-specific effects (geometry × side interactions) | **No.** −0.009, worse in 5/5 repeats. |
| Richer 3D context helps: partner/chain environment features ([environment.py](src/features/environment.py)) | **No** for ranking (−0.018, 1/5), though pooled Pearson +0.02. |
| Wild-type/mutant identity descriptors help | **Inconsistent** (+0.010, 4/5) and they hurt pooled Pearson (0.45 → 0.37). |
| The worst errors are label noise or mapping errors | **Mostly no** (§6.4, point 4). |

No variant gave a consistent paired gain, so none was carried to model C. The geometry set is close to what this data supports for ranking with a low-capacity model.

### 6.6 Adding multi-point rows to training
272 uncensored multi-point rows (29 complexes) added to the 668 single-point rows; per-site features of C are pooled over the sites (sum or mean).
[scripts/14_multipoint_experiment.py](scripts/14_multipoint_experiment.py), [results/multipoint/](results/multipoint/). Per-complex Spearman:

| Training set (C ridge) | Single-point test rows | Multi-point test rows | All test rows | Paired vs singles-only, all rows |
|---|---|---|---|---|
| singles only (= C), sum / mean pooling | 0.460 / 0.460 | 0.053 / 0.119 | 0.332 / 0.329 | – |
| + multi, sum / mean pooling | 0.385 / 0.397 | 0.090 / 0.143 | 0.271 / 0.273 | −0.061 (0/5) / −0.059 (0/5) |
| + multi, 1/√n weights, sum / mean pooling | 0.396 / 0.410 | 0.039 / 0.082 | 0.276 / 0.289 | −0.056 (1/5) / −0.043 (0/5) |

- **Adding multi-point rows hurts single-point ranking** (19 of 20 paired comparisons) and the combined score. Complex weighting reduces but does not remove the harm.
- **Multi-point ΔΔG is hard to predict from per-site features** (0.05–0.14; RMSE 2.0–2.7): the sites interact, and per-site features cannot see that. Mean pooling beats sum pooling, i.e. strict additivity over-predicts large sets.

### 6.7 Adding censored rows with a Tobit-style loss
Of 74 censored single-point rows, 24 unique mutations have a usable lower bound ΔΔG ≥ L (mean bound 3.6 kcal/mol, 10 complexes). A censored row costs
Huber(max(0, L − prediction)): a prediction at or above the bound is free. Censored rows are used in training only.
[scripts/16_censored_experiment.py](scripts/16_censored_experiment.py), [results/censored/](results/censored/).

| Training set (C ridge) | Per-complex Spearman | Paired vs C | RMSE | Mean prediction for rows with ΔΔG > 2 (true 3.45) |
|---|---|---|---|---|
| exact rows only (C) | 0.460 | – | 1.379 | 1.34 |
| + censored, one-sided Huber (Tobit) | 0.455 | −0.005 (2/5) | 1.377 | 1.41 |
| + censored, bound used as exact label | 0.454 | −0.006 (1/5) | 1.377 | 1.41 |

- **No gain.** On held-out censored rows the model predicts below the bound 83–92% of the time, where the one-sided loss equals the exact-label loss. 24 rows are too few to correct the tail.
- **Multi-point + censored together** ([scripts/18_censored_multi_experiment.py](scripts/18_censored_multi_experiment.py), [results/censored_multi/](results/censored_multi/)): still below C on single-point rows (−0.036, 0/5 with mean pooling; −0.010, 2/5 with 1/√n weights).

### 6.8 Combining multi-point and censored rows
[scripts/17_censored_multi_features.py](scripts/17_censored_multi_features.py), [scripts/18_censored_multi_experiment.py](scripts/18_censored_multi_experiment.py),
[results/censored_multi/](results/censored_multi/). Training set: 668 exact single-point + 272 exact multi-point rows, plus the 47 right-censored rows with a numeric lower bound
(24 single-point, 23 multi-point; the 23 add 99 sites), with the one-sided Huber loss in training only. Test scoring is on exact rows. Per-complex Spearman, mean over 5 fold assignments.
(The "+ multi" rows differ by about 0.005 from §6.6 because the extra complexes that only have censored rows change the random fold assignment of the multi-only complexes.)

| Training set | Pooling | Single-point test rows | Multi-point test rows | All test rows | Paired vs singles-only, all rows |
|---|---|---|---|---|---|
| singles only (= C) | mean | **0.460** | 0.119 | **0.329** | – |
| + multi | mean | 0.392 | 0.140 | 0.267 | −0.062 (0/5) |
| + multi + censored | mean | 0.424 | **0.152** | 0.302 | −0.027 (0/5) |
| + multi + censored, 1/√n weights | mean | 0.450 | 0.119 | 0.323 | −0.006 (3/5) |
| singles only | sum | 0.460 | 0.053 | 0.332 | – |
| + multi | sum | 0.397 | 0.083 | 0.282 | −0.050 (0/5) |
| + multi + censored | sum | 0.408 | 0.120 | 0.295 | −0.037 (1/5) |
| + multi + censored, 1/√n weights | sum | 0.392 | 0.087 | 0.281 | −0.051 (1/5) |

1. **Censored rows repair most of the damage that multi-point rows do** (mean pooling, all rows: 0.267 → 0.302 → 0.323), and with 1/√n weights the model ties plain C (−0.006, not
   worse in 3/5 repeats). But **none of the combinations beats C**; the best one only matches it.
2. **Multi-point rows are predicted better when censored rows are included** (sum pooling: 0.083 → 0.120, better than singles-only in 5/5 repeats), but this comes at a cost for single-point rows.
3. **Conclusion for the "more data" branch:** with per-site features and a single linear model, neither multi-point nor censored rows improve model C. The remaining ideas are a model that represents the
   interaction between sites (for multi-point), and a different kind of data (non-antibody SKEMPI pretraining).

## 7. Training diagnostics
Full report: [reports/training_diagnostics.md](reports/training_diagnostics.md). Key points:

- **The optimiser converges.** Huber ridge is convex; all 510 fits of every tuning grid on every outer training set report success (at most 645 L-BFGS iterations, limit 2,000). [convergence.png](results/ablation_ABC/convergence.png)
- **Boosting length.** For C, held-out loss flattens by about round 150 and the number of rounds barely matters. For sequence-only A it is still falling slowly at round 500: A is under-fitted, not over-fitted.
- **Tuning-grid edges.** Every run writes `grid_edges.csv` and warns when a hyperparameter sits on a grid edge in ≥50% of folds. Two such cases in the geometry-only models were fixed by widening the
  grids; no result changed by more than 0.005. A LightGBM still sits on its top edges (PCA 64, 500 rounds) after one widening that gained +0.02; we stopped there and record it as a limitation.
- **Differences between folds are much larger than any effect of training length**, which is why everything is repeated over 5 fold assignments.
- **Learning curve** ([learning_curve.png](results/learning_curve/learning_curve.png)): models trained on 25 / 50 / 75 / 100% of the training complexes, fixed hyperparameters.

  | Training complexes per fold | 9 | 18 | 28 | 37 (all) |
  |---|---|---|---|---|
  | C: augmented ridge | 0.40 | 0.44 | 0.45 | 0.46 |
  | geometry only: ridge | 0.38 | 0.42 | 0.43 | 0.43 |
  | A: LightGBM (sequence only) | 0.14 | 0.17 | 0.26 | 0.31 |

  The geometry models have **plateaued** (train 0.46 vs held-out 0.43): they are limited by their features, not by the amount of data. The sequence-only model is
  **data-limited**: still rising steeply, with a large train/test gap (0.87 vs 0.31).

## 8. Reproducing

Python 3.9. The pinned environment is in [requirements.txt](requirements.txt) (torch is the CPU build):

```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Data: `data/skempi_v2.csv` and `data/SKEMPI2_PDBs/PDBs/` from the SKEMPI 2.0 website. ESM-IF1 features are cached in `data/processed/esmif1.parquet`; recomputing
them needs [requirements-esmif1.txt](requirements-esmif1.txt) on Python ≥ 3.10 (we used Colab).

Hardware: Windows 11, 16-thread CPU, no GPU; a Colab T4 GPU for ESM-2 650M and ESM-IF1.

| Step | Command | Runtime |
|---|---|---|
| clean + dedup, folds | `python scripts/01_clean.py`, `python scripts/02_split.py` | seconds |
| EDA figures | `python scripts/eda_basic.py`, `python scripts/eda_features.py` | seconds |
| geometry features | `python scripts/03_geometry.py` | ~2 min |
| ESM-2 35M features | `python scripts/04_esm2.py` | ~4 min |
| ESM-2 650M + ESM-IF1 features | `python scripts/make_colab_bundle.py`, then [notebooks/02_colab_embeddings.ipynb](notebooks/02_colab_embeddings.ipynb) on Colab | ~15–20 min |
| model A (130 runs) | `python scripts/05_train_A.py 35M 5 3` | ~12 min |
| ablation A/B/C (85 runs) | `python scripts/06_ablation_ABC.py 35M 5 3` | ~22 min |
| ablation D/E (90 runs) | `python scripts/09_ablation_DE.py 35M 5 3` | ~13 h (much slower than A/B/C; cause not diagnosed) |
| diagnostics | `python scripts/diag_training_curves.py`, `python scripts/diag_convergence.py` | ~5 min |
| learning curve (750 fits) | `python scripts/07_learning_curve.py` | ~2 min |
| censored multi-point features (§6.8) | `python scripts/17_censored_multi_features.py` | ~1 min |
| multi-point + censored experiment (40 runs) | `python scripts/18_censored_multi_experiment.py` | ~20 min on 15 workers |
| error analysis | `python scripts/10_error_analysis.py`, `python scripts/19_calibration_case_studies.py` | seconds (script 19 not timed) |
| targeted experiments (§6.5) | `python scripts/11_environment_features.py`, `python scripts/12_hypotheses.py geometry` | ~2 min |
| multi-point (§6.6) | `python scripts/13_multipoint_features.py`, `python scripts/14_multipoint_experiment.py` | ~18 min |
| censored (§6.7) | `python scripts/15_censored_features.py`, `python scripts/16_censored_experiment.py` | ~9 min |
| multi-point + censored (§6.7) | `python scripts/17_censored_multi_features.py`, `python scripts/18_censored_multi_experiment.py` | ~20 min |

Training runs use 15 worker processes.

## 9. Limitations and next steps

**Limitations**
- **Predictions are compressed:** hotspots are under-predicted and improving mutations are not found (§6.4). For lead optimisation this is the most serious gap.
- **Only 20 complexes support the primary metric.** CIs are about ±0.13, so only differences that are consistent across paired repeats are claimed.
- **Wild-type structure only.** Conformational change on mutation is not modelled.
- **ESM-2 35M embeddings only.** The 650M zero-shot score is uninformative (§6.5), but a trained model on 650M embeddings was not tested. The sequence-only LightGBM is under-tuned on the high side (§7).
- **Antibody chains** are assigned from SKEMPI protein names, not by numbering with ANARCI; CDR vs framework annotation is missing.
- **Fusion is concatenation.** A learned fusion (gated MLP) was not compared, on the small-data argument in §4.2.

**Next steps, and why**
1. **Antibody side first:** CDR/framework annotation and an antibody-specific model (AbLang2, AntiFold) for antibody-side rows. Why: it is the largest stratified gap (0.35 vs 0.60) on 58% of the rows.
2. **Transfer from the non-antibody part of SKEMPI** for the sequence branch. Why: the learning curve shows it is the only data-limited component (§7).
3. **Model site interactions for multi-point mutations** (an `is_multi` indicator, a pairwise term between sites). Why: per-site pooling fails on them (§6.6), and they are 30% of the data.
4. **Not** more hand-crafted geometry or ESM-IF1 features: both were tested and the geometry models have plateaued (§6.3, §6.5, §7).
