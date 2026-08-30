# Pre-Registration

**Study:** Inherited or Learned? Decomposing the Contribution of ImageNet Pretraining to Corruption Robustness in Medical Image Classification

**Status:** Committed BEFORE any experimental run. Any deviation from this document must be recorded in `DEVIATIONS.md` with a date, a reason, and the commit hash at which it was decided.

**Date committed:** August 30, 2026
**Commit hash of this file:** 5b1f1b4

---

## 1. Research question

To what extent is the corruption robustness of medical image classifiers on MedMNIST-C inherited from ImageNet pretraining rather than learned from medical training data, and does that inherited robustness depend on corruption family, corruption severity, and training-set size?

## 2. Motivation and gap

Di Salvo, Doerrich & Ledig (MedMNIST-C, ADSMI@MICCAI 2024, arXiv:2406.17536) initialized every benchmarked network with ImageNet weights and included no randomly-initialized arm. Doerrich et al. (Sci Rep 15:7669, 2025) likewise start every training scheme from pretrained weights. Consequently, published MedMNIST-C robustness numbers confound robustness learned from medical data with robustness inherited from ImageNet. The MedMNIST-C authors additionally hypothesize, without testing, that "the training set size might play an important role" (§3, Experimental results).

## 3. Hypotheses

**H1 (primary).** ImageNet initialization reduces corrupted-set balanced error relative to random initialization, but the _magnitude_ of the benefit differs across corruption families. Specifically, the benefit is smaller for **task-specific** corruptions (stain deposit, bubble, black corner, characters) than for **noise, blur, and digital** corruptions.

**H2.** On BloodMNIST, the accuracy benefit of ImageNet initialization increases as the training fraction decreases (100% → 50% → 25% → 10%).

**H3.** Pretraining improves absolute corrupted-set accuracy more than it improves relative robustness; i.e. much of the corrupted-set gap is accounted for by the clean-accuracy gap.

**H4.** The pretraining benefit, measured as the balanced-accuracy gap, decreases monotonically as corruption severity increases from 1 to 5.

**Exploratory (not confirmatory, will be labelled as such):** whether the H1 pattern is consistent across the three modalities tested. With three datasets this cannot support a strong modality claim.

## 4. Datasets

| Dataset        | Modality              | Task            | Train / Val / Test     | Corruptions | Task-specific present          |
| -------------- | --------------------- | --------------- | ---------------------- | ----------- | ------------------------------ |
| BloodMNIST     | Blood cell microscopy | multi-class (8) | 11,959 / 1,712 / 3,421 | 11          | yes (stain_deposit, bubble)    |
| DermaMNIST     | Dermatoscope          | multi-class (7) | 7,007 / 1,003 / 2,005  | 15          | yes (black_corner, characters) |
| PneumoniaMNIST | Chest X-ray           | binary (2)      | 4,708 / 524 / 624      | 13          | **no**                         |

Clean data: MedMNIST+ at 224×224 (https://zenodo.org/records/10519652). Corrupted test sets: MedMNIST-C (https://zenodo.org/records/11471504). Official predefined splits only; no re-splitting.

PneumoniaMNIST is included deliberately as a control: it has no task-specific corruptions, so under H1 it should show a comparatively _uniform_ pretraining benefit.

Licenses: CC BY 4.0, except DermaMNIST / DermaMNIST-C which are CC BY-NC 4.0. Known label-quality concerns for DermaMNIST (Abhishek et al., Scientific Data 2025) will be cited in Limitations.

## 5. Model and initialization conditions

torchvision ResNet-18, final layer replaced to match n_classes.

- **Condition A (pretrained):** `IMAGENET1K_V1` weights
- **Condition B (scratch):** `weights=None`, default PyTorch random initialization

Initialization is the **only** manipulated variable. Architecture, optimizer, schedule, batch size, epoch budget, early-stopping rule, augmentation, and input normalization are identical across arms.

## 6. Training protocol

AdamW, cosine annealing (single cycle), batch size 64, input 224×224, maximum 50 epochs, early stopping with patience 10 on validation balanced accuracy, best-validation checkpoint used for testing. Augmentation: resize and normalize only. **No corruption-based augmentation** — that is a different research question and its inclusion would invalidate this design.

**Learning rate.** Pretrained arm: 1e-4 (matching both parent papers). Scratch arm: selected once by searching {1e-4, 3e-4, 1e-3} on BloodMNIST validation with seed 0 only, then **fixed for all scratch runs across all datasets**. Rationale: 1e-4 AdamW is a fine-tuning learning rate; granting the scratch arm a small search prevents an artificially weakened baseline. The search results will be reported in full.

## 7. Seeds and experiment matrix

Seeds: 0, 1, 2, 3, 4 for the core; 0, 1, 2 for the ablation.

| ID  | Dataset        | Init | Train fraction | Seeds | Runs |
| --- | -------------- | ---- | -------------- | ----- | ---- |
| E1  | BloodMNIST     | A, B | 100%           | 5     | 10   |
| E2  | DermaMNIST     | A, B | 100%           | 5     | 10   |
| E3  | PneumoniaMNIST | A, B | 100%           | 5     | 10   |
| E4  | BloodMNIST     | A, B | 10%, 25%, 50%  | 3     | 18   |

Total 48 runs. Subsampling for E4 is class-stratified and seeded; the same subsample is used by both arms at a given (fraction, seed).

## 8. Metrics

- Clean balanced accuracy
- Corrupted balanced accuracy, per corruption and per severity
- **Absolute drop** = clean balanced accuracy − mean corrupted balanced accuracy
- **BE** (AlexNet-normalized balanced error) — **primary normalized metric**
- **rBE** (relative balanced error) — secondary, subject to the exclusion rule in §9
- Family-level aggregates over {digital, noise, blur, color, task-specific}
- **Δ-robustness** = metric(scratch) − metric(pretrained), per corruption and per family

Metrics computed with the official `medmnistc.eval.Evaluator`, extended to also dump per-severity errors.

## 9. Known metric hazard and its handling (declared in advance)

For PneumoniaMNIST, the rBE denominator `alexnet_error_c − alexnet_clean_score` is **negative** for `contrast_up` (−0.0056) and `gamma_corr_up` (−0.0048), making rBE uninterpretable for those corruptions. Additional corruptions have denominators below 0.05 and yield unstable rBE.

Pre-registered rules:

1. BE is the primary normalized metric.
2. Corruptions with a negative denominator are **excluded from all rBE aggregates**, and the exclusion is reported in the paper with an appendix table of all denominators.
3. Corruptions with denominator < 0.05 are flagged in the appendix.
4. Absolute drop is always reported alongside BE and rBE.

## 10. Primary endpoint

The **interaction between initialization and corruption family**: the difference in Δ-robustness (BE) between the task-specific family and the pooled {noise, blur, digital} families, on the two datasets that contain task-specific corruptions (BloodMNIST, DermaMNIST).

## 11. Secondary endpoints

- Clean and corrupted balanced accuracy per dataset and arm (H3)
- Balanced-accuracy gap as a function of training fraction (H2)
- Balanced-accuracy gap as a function of severity (H4)
- Correlation between clean-accuracy gain and robustness gain (H3)

## 12. Statistical analysis

- All results reported as mean ± SD over seeds. No single-seed number appears in the paper.
- Unit of pairing for significance tests is the **corruption type**, not the seed.
- Two-tailed **Wilcoxon signed-rank** tests on per-corruption Δ-BE, within dataset.
- **Bootstrap 95% CIs** (10,000 resamples) for family-level Δ.
- **Holm–Bonferroni** correction across the family-level tests within each dataset.
- α = 0.05.
- H2 and H4 trends: report direction, effect size, and CIs. No regression p-values will be computed on 3–5 points.

## 13. Decision rules

| Hypothesis | Supported if                                                                                                                                                                                         | Rejected if                                                                       |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| H1         | Δ-BE for task-specific corruptions is significantly smaller than for pooled noise/blur/digital, on at least one of BloodMNIST/DermaMNIST after Holm correction, with the same direction on the other | No significant family interaction on either dataset, or the direction is reversed |
| H2         | Balanced-accuracy gap increases monotonically as fraction decreases, with non-overlapping CIs between 10% and 100%                                                                                   | Non-monotonic or flat within CIs                                                  |
| H3         | Δ in absolute corrupted accuracy substantially exceeds Δ in rBE; clean-gain vs robustness-gain correlation is strongly positive                                                                      | Robustness gain is independent of clean gain                                      |
| H4         | Gap decreases monotonically across severities 1–5 in the majority of dataset × family cells                                                                                                          | No consistent monotonic trend                                                     |

A rejected hypothesis will be reported as rejected. Null and negative results are reported with the same prominence as positive ones.

## 14. Exclusion criteria

- A run is excluded only if it **fails to train** (validation balanced accuracy at or below chance at the end of the epoch budget). Exclusions are logged with the seed and reported in the paper; a failing configuration is reported as a finding, not silently dropped.
- Corruptions are excluded from rBE aggregates only under the §9 denominator rule.
- No other exclusions. Individual seeds are never dropped for being outliers.

## 15. Failure handling

- **Scratch arm fails to converge on a dataset:** report it as a result. Prior work (SegSTRONG-C) reports comparable failures. Do not tune the pretrained arm down to compensate.
- **Compute shortfall:** cut in this pre-declared order — (1) qualitative figure, (2) H4 severity analysis, (3) E4 ablation, (4) DermaMNIST, (5) seeds 5 → 3. Cuts are recorded in `DEVIATIONS.md`.
- **Session loss:** checkpoints and per-epoch logs are written continuously; runs resume rather than restart.

## 16. Analysis plan and anti-cherry-picking commitments

1. The analysis script is written against **synthetic placeholder data before real results exist**, and committed.
2. Results are consolidated into a single tidy CSV; all tables and figures are generated from that CSV by script, never by hand.
3. No hypothesis is added, removed, or reworded after results are seen. Any post-hoc observation is labelled **exploratory** in the paper.
4. No metric is added after results are seen unless labelled exploratory.
5. All 48 runs appear in the released results, including failures.
6. The full results CSV is released with the paper.

## 17. What is explicitly NOT claimed

- Not a claim that pretraining improving corruption robustness is a new finding (Hendrycks, Lee & Mazeika, ICML 2019).
- Not a claim about clinical performance or deployment. These are downsampled 224×224 research datasets with synthetic corruptions.
- Not a causal claim about _why_ pretraining transfers. Mechanistic language will be framed as hypothesis, not conclusion, unless a representational analysis is performed.
- Not a generalization to modalities, architectures, or corruption types outside those tested.
