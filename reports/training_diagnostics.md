# Training diagnostics: does training converge, is tuning adequate, and what limits the models?

This report collects every check we ran on the training process of the ΔΔG models, what each check showed, and what we changed
because of it. The model results themselves are in the [README](../README.md) (§6). Here the question is whether those results come from
models that were **trained properly**: optimisers that reached their minimum, boosting stopped at a sensible point, hyperparameter grids wide
enough, and a clear picture of what limits each model.

**Summary**
1. **Optimisation converges.** Every Huber-ridge fit reaches the minimum of its (convex) objective. All 860 fits checked succeed. In the 510 where
   we also measured the gradient, it shrinks by ≥770×, and the objective settles to a relative change of ~1e-12.
2. **Boosting length matters little, and is now tuned on inner folds.** Sequence-only LightGBM was mildly overfitting at a fixed 300 rounds (held-out
   loss +4% vs its best round). Once structure is added, the held-out loss is flat after about 150 rounds, so the exact number barely matters.
3. **Two tuning grids were too narrow; fixing them changed one result.** The geometry-only grids were fixed with no effect on results (≤0.005). The sequence-only
   grid was widened twice and A improved from 0.28 to 0.30 per-complex Spearman. A still sits at the top of its grid, which we report as a limitation.
4. **Which complexes land in the test fold matters far more than any training setting.** Held-out loss differs by about 2× between folds, while
   training length changes it by about 4%. This is why every result is repeated over 5 fold assignments.
5. **The geometry models are limited by their features, the sequence model by data.** Geometry-only performance has plateaued with train ≈ test.
   Sequence-only performance is still rising steeply, with a large train/test gap. This sets the priorities for the next steps.

---

## 1. Setting

- **Task:** regression of ΔΔG (kcal/mol) on 668 single-point mutations from 46 antibody–antigen complexes (SKEMPI 2.0).
- **Models:** frozen encoders feed fixed features into one of two regressors:
  - **augmented ridge:** one linear layer, Huber or squared loss, L2 penalty α, with the zero-shot ESM-2 score penalised less ("boost"), and optional PCA of the embeddings
  - **LightGBM:** shallow boosted trees on PCA-compressed embeddings plus scalar features
- **Feature sets** (the ablation ladder): A = ESM-2 35M sequence features; B = A + distance to the partner; C = A + 10 structural geometry features;
  geometry only = the 10 geometry features alone.
- **Evaluation:** grouped CV (no complex in both train and test), repeated over **5 fold assignments**, with **3 seeds** for LightGBM. Hyperparameters are
  chosen by an inner grouped CV inside each training fold, scored by pooled Spearman.
  Headline metric: per-complex Spearman (mean over the 20 complexes with ≥10 test rows).
- **One rule for every diagnostic below:** curves on the *outer* test folds are only looked at, **never used to choose anything**. Every choice is made on
  inner folds, otherwise test information would leak into training.

What "convergence" means depends on the model, so we split it into three questions:

| Question | Applies to | Check |
|---|---|---|
| Q1. Did the optimiser reach the minimum of its objective? | Huber ridge (iterative L-BFGS) | success flag, iterations, final gradient, objective curve |
| Q2. When does more training stop helping? | LightGBM (boosting never "finishes") | training vs held-out loss per round, rounds chosen by inner CV |
| Q3. Was the hyperparameter search wide enough? | all tuned models | how often inner CV picks a grid edge |
| Q4. Is the model limited by data or by features? | the main models | learning curve over training-set size |

MSE ridge is solved in closed form, so Q1 doesn't apply to it. The baselines (training mean, calibrated zero-shot score) are one-step fits.

---

## 2. Q1: does the optimiser converge? (Huber ridge)

**Why check it.** The Huber-loss ridge is fitted with an iterative optimiser (L-BFGS, at most 2,000 iterations). Its objective is **convex**, so there
is exactly one minimum and no local optima. "Converged" therefore has a precise meaning: the optimiser reached that minimum. If it stopped early,
differences between models could reflect the optimiser rather than the features.

**What we measured**
- the optimiser's own success flag and its iteration count
- the **objective per iteration**, as the relative gap to the final value
- the **final gradient**. At the minimum the gradient is exactly zero, so we report max |gradient| at the end relative to its value at the start. This is the
  stricter test: the objective curve alone only shows that the optimiser stopped changing, not that it stopped at the minimum.

**Scope.** Not just the configurations that were finally chosen: **every configuration of every tuning grid, refit on every outer training set**
of the primary split. That is 350 fits for model A ([diag_training_curves.py](../scripts/diag_training_curves.py): success flag and iterations; all 350
converged, at most 617 iterations) and 510 for the ablation models ([diag_convergence.py](../scripts/diag_convergence.py): also the gradient, table below).

**Result**

| Model | Fits | Converged | Max iterations | Gradient reduction (worst / median) |
|---|---|---|---|---|
| A: augmented ridge | 150 | 150 | 580 | 780× / 28,000× |
| B: augmented ridge | 150 | 150 | 569 | 1,240× / 30,000× |
| C: augmented ridge | 150 | 150 | 645 | 950× / 23,000× |
| geometry only: ridge | 30 | 30 | 27 | 13,500× / 62,000× |
| location + substitution | 30 | 30 | 19 | 25,600× / 110,000× |

For the chosen configurations, the objective falls to ~1e-12 of its final value within 10–65 iterations ([convergence.png](../results/ablation_ABC/convergence.png),
bottom row; [training_curves_ridge.png](../results/model_A/training_curves_ridge.png)).

**A problem this check exposed.** When we widened the penalty grid down to α = 0.01 (§4), the full-embedding models (961 features, ~530 rows) became
ill-conditioned: α = 0.01 needed ~1,900 iterations (8.9 s) per fit instead of ~60 (0.4 s). The run grew past our 30-minute job limit and was stopped. Those
models never chose a small α in the first place (they pick α ≈ 100), so low penalties are now offered only to the low-dimensional models. We also
found that recording the objective history cost one extra objective evaluation per iteration; it is now switched on only in the diagnostic scripts.

**Conclusion.** Optimisation is not a source of error in any reported ridge result.

---

## 3. Q2: when does more boosting stop helping? (LightGBM)

**Why check it.** Gradient boosting adds one tree per round and keeps lowering the training loss indefinitely; given enough rounds it memorises
the training rows. So "has the training loss converged?" is the wrong question. The right one is **when the held-out loss stops improving**. Training
past that point fits noise. Stopping before it leaves signal unused.

**What we measured.** For each outer fold, we refit with the hyperparameters chosen in that fold and recorded the Huber loss after every round, on the training rows and
on the held-out fold. In the ablation version, curves are refit to 500 rounds and the round chosen by inner CV is marked.

### 3.1 Model A with a fixed 300 rounds
([training_curves_lgbm.png](../results/model_A/training_curves_lgbm.png); 4 panels = Huber/MSE × unweighted/weighted)

| Loss | Best held-out round (median over folds) | Held-out loss: at best round → at round 300 |
|---|---|---|
| Huber | 133–160 | 0.618 → 0.641 (+4%) |
| MSE | 296–299 | 1.88 → 1.96 (+4%) |

- With Huber, held-out loss flattened around round 130 and then rose slightly. In 3 of 5 folds it started rising after 50–100 rounds while training loss kept
  falling. That is **mild overfitting**.
- With MSE the mean held-out curve was still drifting down at 300. That was driven by the two hard folds; the other three were already rising.
- **Action:** the number of rounds became a tuned hyperparameter, chosen on inner folds ({100, 200, 300}, later widened to {25 … 500}, §4).

### 3.2 After tuning: A vs C
([convergence.png](../results/ablation_ABC/convergence.png), top row)

| Fold | A: chosen round | A: held-out best round | C: chosen round | C: held-out best round |
|---|---|---|---|---|
| 0 | 500 | 120 | 500 | 126 |
| 1 | 500 | 497 | 200 | 392 |
| 2 | 300 | 498 | 300 | 119 |
| 3 | 500 | 168 | 100 | 193 |
| 4 | 300 | 499 | 100 | 500 |

- **A (sequence only) is a slow learner.** The mean held-out loss is still falling at round 500, and inner CV chose 300–500 in every fold. With 961
  noisy embedding features and ~530 rows, each tree extracts little signal, so A needs many rounds. It is **under-fitted rather than over-fitted**.
- **C (+ geometry) has a flat optimum.** Held-out loss flattens by about round 150 and stays flat to 500. Inner CV's choices spread over 100–500, which
  is what a flat optimum looks like: the number of rounds barely matters for C.
- **Inner CV and the held-out optimum often disagree.** Two reasons: inner CV ranks by Spearman while the curves show Huber loss, and an inner fold
  holds only a handful of complexes. Because the curves are flat, the cost of a "wrong" choice is small (except A, which keeps improving).

### 3.3 Folds matter more than training length
Across folds, held-out loss ranges from about **0.36 to 0.80**. Training length changes it by about **4%**. Two folds are consistently hard; they contain
the complexes with the largest ΔΔG values (e.g. the destabilising tail of 3HFM). This is the main reason every result is reported over **5 fold
assignments** with mean ± std, plus a bootstrap CI over complexes (about ±0.13 for per-complex Spearman), rather than from a single split.

**Conclusion.** Boosting length is now tuned without touching the test folds. It matters for sequence-only A, and hardly at all once structure is in the model.

---

## 4. Q3: were the hyperparameter grids wide enough?

**Why check it.** If inner CV keeps picking the lowest or highest value of a grid, the best setting probably lies **outside** the grid, and the model
is under-tuned. That matters most when the model is half of a comparison. "Does sequence add anything on top of geometry?" (C vs geometry-only) is only fair
if both sides are tuned properly.

**What we measured.** For every tuned hyperparameter with ≥3 grid values, the share of picks at the lowest or highest value, over all outer folds × fold
assignments × seeds. Every run now writes [grid_edges.csv](../results/ablation_ABC/grid_edges.csv) and prints a warning at ≥50%. (Grids with only two values, such as
leaves {4, 7}, are not judged: every pick there is an edge.)

**Three rounds of fixes**

| Run | Problem found | Change | Effect on results |
|---|---|---|---|
| 1 | geometry-only ridge picked the lowest α (1) in **76%** of folds; geometry-only LightGBM the fewest rounds (100) in **81%** | α down to 0.01 (low-dimensional models only); rounds from 25 | edges gone (32%, 4%); **no metric moved by more than 0.005** |
| 2 | sequence-only A LightGBM picked the largest PCA size (32) in **81%** and the most rounds (300) in **61%** | PCA 64 and 500 rounds added, for A, B and C alike | A: **0.28 → 0.30** per-complex Spearman, pooled Pearson 0.33 → 0.40; B − A gap shrank 0.06 → 0.04, C − A 0.15 → 0.13 |
| 3 | A still at the top: 64 components in **81%**, 500 rounds in **71%**; B's PCA size at 57% | none (stopped) | – |

**Why we stopped widening at run 3.**
- A is a reference rung of the ablation, not a candidate final model.
- The last widening gained only +0.02.
- The training curve (§3.2) shows A still improving slowly with no sign of a sharp optimum just beyond the grid.

We report it as a limitation: a sequence-only model on these embeddings may need more capacity than we gave it, so the sequence-vs-structure gaps are, if anything,
slightly overstated. C and both geometry-only models are not at any edge, so the C vs geometry-only comparison is fair.

**Conclusion.** The final ablation numbers come from adequately tuned models, except sequence-only A, which is marked as possibly under-tuned. None of
the ablation's conclusions changed through these fixes: every paired comparison kept its direction in all 5 fold assignments.

---

## 5. Q4: is each model limited by data or by features? (learning curve)

**Why check it.** A model that has stopped converging *with respect to data* (more examples no longer help) needs better features. A model still
improving with more data would benefit from more, or related, training data. That distinction decides what to do next.

**What we measured** ([07_learning_curve.py](../scripts/07_learning_curve.py), [learning_curve.png](../results/learning_curve/learning_curve.png)):
- **Subsets:** in every outer fold of all 5 fold assignments, each model was trained on a random 25 / 50 / 75 / 100% of the training **complexes**. Whole
  complexes only, 3 draws per fraction, the same draws for every model. Scoring was on the untouched test fold: 750 fits.
- **Fixed hyperparameters:** each model used its most frequently chosen setting from the ablation, not re-tuned at each size. Re-tuning would have cost ~40,000 fits.
  If anything this makes the smallest sizes slightly pessimistic.

| Training complexes per fold | 9 | 18 | 28 | 37 (all) | Train per-complex ρ at 37 |
|---|---|---|---|---|---|
| C: augmented ridge | 0.40 | 0.44 | 0.45 | **0.46** | 0.56 |
| geometry only: ridge | 0.38 | 0.42 | 0.43 | **0.43** | 0.46 |
| A: LightGBM (sequence only) | 0.14 | 0.17 | 0.26 | **0.31** | 0.87 |

(held-out per-complex Spearman, mean over fold assignments and draws)

1. **Geometry-only has plateaued and is feature-limited.** It gains +0.003 from 75% to 100%, and its training and held-out scores nearly coincide
   (0.46 vs 0.43). It is not overfitting: the 10 geometry features hold no more signal for it to extract. **More complexes of the same kind will not
   help it; richer structural features will.**
2. **Sequence-only A is data-limited.** It rises steeply with no plateau, and its training score (0.87) is far above its held-out score (0.31): high
   variance. At full size its RMSE (1.36) already equals C's. It is the component that would gain most from more data.
3. **C is almost flat at the end** (+0.003 from 75% to 100%) with a moderate train/test gap (0.56 vs 0.46). Its only still-improving part is the sequence
   branch, so the fusion gain (+0.02 per-complex today) could grow with more data.

---

## 6. Related checks made along the way

- **Loss function (model A, 5 fold assignments).** Huber beats MSE on per-complex Spearman in **10/10** paired comparisons (+0.025). MSE wins on pooled
  Pearson and RMSE, because it chases the large destabilising values. The training curves show the same split: under MSE, the few large errors keep pulling
  the model for longer. Huber is the default because the primary metric is per-complex.
- **Sample weighting.** 1/√(rows per complex) vs unweighted gave no consistent benefit: it helped ridge in 8/10 comparisons (+0.02), LightGBM in 4/10 and
  location + substitution in 2/10. Unweighted is the default.
- **Why the embedding models need regularisation at all.** Trained on 960 columns of pure random noise with our real labels, plain linear
  regression and LightGBM both reach training Spearman 1.00 and test ~0.07. Ridge refuses to memorise (training 0.71). This is why ridge keeps the full
  embeddings while LightGBM gets PCA.
- **Variance across fold assignments.** The std of per-complex Spearman across the 5 fold assignments is 0.01–0.05 depending on the model. That is much
  smaller than the bootstrap CI over complexes (about ±0.13), which reflects that only 20 complexes support the metric. Model comparisons are
  therefore made with **paired differences on identical folds**, not by overlapping intervals.

---

## 7. What we conclude, and what it changes

| Finding | Consequence |
|---|---|
| Ridge optimisation converges in every fit | Reported ridge results reflect the features, not the optimiser |
| Boosting length is tuned on inner folds; flat optimum once structure is added | No further work needed on training length |
| Geometry-only and C are not at grid edges; their comparison is fair | The claim "sequence adds +0.02 on top of geometry" stands (5/5 fold assignments) |
| Sequence-only A is under-fitted, possibly under-tuned, and data-limited | Treat A's numbers as a lower bound; the sequence branch is where more data would help |
| Geometry is feature-limited | **Next: better structural features**, namely ESM-IF1 scores in complex vs unbound (model D) and its embedding (model E) |
| The sequence branch is data-limited | **Later: transfer learning**, pretraining on the non-antibody protein–protein part of SKEMPI before fitting on antibody–antigen data |
| Fold-to-fold differences dominate | Keep 5 repeated fold assignments and paired comparisons for every future model |

## 8. Limitations of these diagnostics

- Training curves and the gradient check use the **primary split** (fold assignment 0) and seed 0. The other fold assignments are covered by the grid-edge
  and results tables, not by curves.
- The learning curve uses **fixed hyperparameters**, and extrapolates nothing beyond the 37 training complexes we have.
- Inner-CV selection uses **pooled Spearman**, while the boosting curves show **Huber loss**, so "chosen round" and "best held-out round" are not
  expected to coincide.
- Everything here concerns ESM-2 **35M**. The 650M model may change the sequence-branch conclusions (planned on Colab).

## Reproducing

| Output | Command | Runtime (16-thread CPU) |
|---|---|---|
| model A runs + curves (loss × weighting) | `python scripts/05_train_A.py 35M 5 3`, then `python scripts/diag_training_curves.py` | ~12 min + ~2 min |
| ablation A/B/C with grid-edge report | `python scripts/06_ablation_ABC.py 35M 5 3` | ~22 min |
| convergence diagnostics (A vs C, gradient check) | `python scripts/diag_convergence.py` | ~3 min |
| learning curve | `python scripts/07_learning_curve.py` | ~2 min |

Note: `diag_training_curves.py` reads model A's `chosen_params.json` from the model-A run (loss × weighting grid, fixed 300 rounds), and
`diag_convergence.py` / `07_learning_curve.py` read the ablation run's. Run them after their training script.
