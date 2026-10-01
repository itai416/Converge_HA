# Project Plan: Multimodal ΔΔG Prediction for Antibody–Antigen Mutations

## Goal and agreed framing
- **Task:** regression of ΔΔG = RT·ln(Kd_mut / Kd_wt) (kcal/mol, positive = weaker binding) on the SKEMPI 2.0 AB/AG subset.
- **v1 scope:** single-point, uncensored rows with valid affinity values (770 rows, 676 unique mutations, 47 complexes).
- **Use case we claim:** optimizing antibodies against a **known antigen** (lead optimization). We do not claim generalization to new targets.
- **Split:** group K-fold by complex, stratified by antigen cluster. Test complexes are tagged "seen antigen" or "unseen antigen".
- **Compute:** pretrained-model embeddings are extracted once on Colab and cached. Everything else runs on CPU.

---

## Stage 1: Data cleaning and label construction
1. Filter rows where `Hold_out_type` contains `AB/AG`.
2. Parse `Temperature` (strip "(assumed)"; default 298 K) and compute ΔΔG from the `*_parsed` affinities.
3. Split into three sets: **v1** (single-point, uncensored, valid), **multi-point** and **censored**. The last two are kept for later extensions.
4. **Identify partners:** split `#Pdb` into PDB id, partner-1 chains and partner-2 chains. Decide which partner is the antibody from the chain sequences (ANARCI numbering, or a heavy/light-chain heuristic as fallback), not from the `Protein 1/2` names.
5. **Map residues:** use the `.mapping` files to handle PDB numbering and insertion codes (e.g. `100b`). Assert that the wild-type residue in the PDB matches the mutation string, and log and drop any mismatch.
6. **Deduplicate repeats:** average repeated measurements of the same (complex, mutation). Keep the per-group std as a measure of experimental noise.
7. **Annotate each row:**
   - mutation side (antibody or antigen)
   - for antibody-side mutations: CDR or framework, with IMGT/Chothia region
   - SKEMPI location class (COR/RIM/SUP/INT/SUR)
   - to-Ala flag
   - anti-idiotype flag
8. **Cluster antigens** by sequence identity (pairwise alignment, threshold of about 40%) to form the stratification variable. Anti-idiotype complexes form their own class.
9. **Output:** `data/processed/v1.parquet` plus a data card listing the row counts dropped at each step.

## Stage 2: Exploratory data analysis
- ΔΔG distribution: overall, per complex and per antigen cluster. Look at skew and the share of stabilizing, neutral and destabilizing mutations.
- ΔΔG by location class, mutation side, CDR vs framework, and wild-type/mutant amino acid (substitution heatmap).
- Alanine-scan bias: ΔΔG for to-Ala vs other substitutions.
- **Noise floor** from repeated measurements: std, and an estimated upper bound on achievable correlation.
- Simple features vs ΔΔG (burial, contacts, hydrophobicity change) to check whether there is signal before any modeling.
- **Output:** an EDA notebook whose findings drive the feature and evaluation choices below.

## Stage 3: Split design
- **Primary:** 5-fold StratifiedGroupKFold. The group is the complex and the stratum is the antigen cluster, so no complex appears in both train and test. Fold assignments are saved to `data/processed/folds.csv` with a fixed seed.
- Check fold balance on row counts, antigen mix and ΔΔG distribution. Lysozyme (13 complexes) must be spread across folds.
- **Hyperparameter tuning** uses nested (inner) grouped CV inside each training fold only, so the test fold never influences model selection.
- **Secondary split:** random split within each complex (the "optimize this exact antibody" scenario). It serves as an easier reference point, and the gap between the two splits measures memorization.
- Every test complex is tagged **seen antigen** or **unseen antigen** for reporting.

## Stage 4: Representations for each modality
All features are computed on the **wild-type complex** (no mutant structures are needed). Where it makes sense, features are expressed as **mutant minus wild-type** so that swapping the two negates them.

**Sequence** (Colab, run once per chain and cached):
- ESM-2 650M (ESM-2 35M as a CPU fallback). Features:
  - masked-marginal log-likelihood ratio, log p(mut) − log p(wt), at the mutated site
  - site embedding for wild-type and mutant, plus their difference
  - local window mean
- Optional: AbLang2 log-likelihood ratio for antibody-side mutations only.

**Structure:**
- **ESM-IF1** (inverse folding) run on the full complex backbone. Features: log p(mut) − log p(wt) at the site conditioned on the complex, and the site embedding.
  - **Key signal:** the same log-likelihood ratio with the partner chains removed. Complex minus isolated chain gives an interface-specific score.
- **Hand-crafted geometric features** (CPU, with Biopython/FreeSASA):
  - relative solvent accessibility in the complex and unbound, and ΔSASA on binding
  - number of partner atoms within 4.5 Å and 8 Å, and minimum distance to the partner
  - hydrogen bonds and salt bridges across the interface
  - secondary structure and B-factor
- **Mutation descriptors:** changes in volume, hydrophobicity and charge, and the BLOSUM62 score.

High-dimensional embeddings are **compressed with PCA fit on the training fold only** (8–32 components) to avoid overfitting on about 600 training rows.

## Stage 5: Fusion
Options we will compare (each choice must be justified in the write-up):
1. **Early fusion:** concatenate the compressed sequence and structure blocks with the scalar features, then feed them to the model. This is the main approach, because it is low-capacity and suits small data.
2. **Late fusion:** separate sequence-only and structure-only models, combined by a stacking regressor fit on inner-fold out-of-fold predictions.
3. **Gated fusion MLP**, optional: a small MLP with a learned gate per modality and modality dropout. It is only kept if it beats (1) under nested CV.

The argument for fusion comes from **ablations**: sequence-only vs structure-only vs fused, all evaluated on the same folds.

## Stage 6: Models and training
- **Baselines (required):**
  - global mean, and antigen mean taken from the training fold
  - location class + substitution-type linear model
  - zero-shot ESM-2 log-likelihood ratio and zero-shot ESM-IF1 log-likelihood ratio (no training)
  - hand-crafted features + gradient boosting
- **Main models:** ridge regression and gradient boosting (LightGBM) on the fused features, plus the optional small MLP (PyTorch, CPU).
- **Loss:** Huber, which is robust to experimental outliers. Features are standardized using training-fold statistics.
- **Antisymmetry augmentation:** add reversed mutations with −ΔΔG and negated difference features. Test this as an ablation, since it is only approximate because the structure stays wild-type.
- Early stopping and hyperparameter search happen on the inner folds only. Results are reported over 3–5 seeds.

## Stage 7: Metrics and evaluation
- **Regression:** Pearson, Spearman, RMSE and MAE, computed both **pooled** and **per complex** (averaged, for complexes with at least 10 mutations).
- **Derived classification** for interpretation: 3 classes (below −0.5 = improving, −0.5 to 1 = neutral, above 1 = destabilizing) scored with macro-F1 and balanced accuracy. AUROC for identifying hotspots (ΔΔG > 1) and improving mutations.
- **Uncertainty:** bootstrap 95% confidence intervals resampled over complexes, and paired per-fold comparisons between models.
- **Stratified reporting:** seen vs unseen antigen, antibody-side vs antigen-side mutations, anti-idiotype vs other complexes.
- **Reference points:** the noise-floor ceiling from repeated measurements, and the gap between the primary and secondary splits.
- The "SKEMPI benchmark result" is the primary grouped-CV table on the AB/AG subset.

## Stage 8: Error analysis
- Residuals by location class, mutation side, CDR region, wild-type/mutant amino acid, to-Ala vs other, and true ΔΔG magnitude. We expect regression toward the mean, so large and improving mutations will be under-predicted.
- Per-complex performance vs training-set support (number of rows, whether the antigen was seen).
- Feature importance (SHAP on the gradient-boosting model) and the modality ablation table.
- Case studies of the 5–10 worst errors, checked against the structure: is the error due to label noise, a mechanism the features miss, or a mapping error?
- Calibration: predicted vs observed ΔΔG by bin.

## Stage 9: Extensions (after v1)
- **Multi-point mutations:** combine per-site features with sum pooling, compared against an additivity baseline.
- **Censored rows:** a Tobit / censored Huber loss that only penalizes predictions below the lower bound.
- **Transfer learning:** pretrain on non-antibody SKEMPI protein–protein interaction data, then fine-tune on AB/AG.
- **Better structural context:** mutant structure modeling (e.g. side-chain repacking), or AntiFold/SaProt embeddings.

## Stage 10: Repository and reproducibility
```
README.md                 method, decisions, results, limitations, next steps, hardware and runtime
requirements.txt          pinned versions (Python 3.9-compatible)
configs/                  YAML per experiment
src/data/                 cleaning, residue mapping, antigen clustering, splits
src/features/             sequence, structure and handcrafted feature extractors
src/models/               baselines, fusion models, training loop
src/eval/                 metrics, bootstrap, stratified reports, plots
notebooks/                01_eda, 02_colab_embeddings, 03_error_analysis
scripts/                  one command per stage (clean → features → train → evaluate)
results/                  metrics tables and figures (committed)
prompts/                  AI prompt history
```
- Fixed seeds, fold assignments saved to file, cached embeddings keyed by (complex, chain, site), and runtime logged per stage.
