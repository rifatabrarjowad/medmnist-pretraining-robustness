# Inherited or Learned?
## Decomposing the Contribution of ImageNet Pretraining to Corruption Robustness in Medical Image Classification

[![Status](https://img.shields.io/badge/status-preprint%20manuscript-blue)](#manuscript)
[![Study](https://img.shields.io/badge/study-preregistered-success)](#preregistration-and-deviations)
[![Python](https://img.shields.io/badge/Python-3.x-blue)](#reproducibility)

This repository contains the code, analysis, figures, result tables, and manuscript for a controlled study of how ImageNet initialization affects corruption robustness in medical image classification.

The study compares **ImageNet-pretrained** and **randomly initialized** ResNet-18 models on **BloodMNIST, DermaMNIST, and PneumoniaMNIST**, evaluated using **MedMNIST-C**.

---

## Research question

> To what extent is corruption robustness in medical image classification inherited from ImageNet pretraining rather than learned from the target medical dataset?

The study focuses on four preregistered questions:

1. Does the effect of ImageNet initialization depend on corruption family?
2. Does pretraining become more valuable when training data are limited?
3. Are gains under corruption largely explained by gains on clean data?
4. Does the pretraining advantage decrease systematically as corruption severity increases?

---

## Experimental design

Two ResNet-18 initialization conditions were compared:

- **ImageNet-pretrained:** `IMAGENET1K_V1`
- **Scratch:** `weights=None`

The architecture, official dataset splits, preprocessing, optimizer family, learning-rate schedule, stopping criteria, model-selection procedure, and evaluation pipeline were matched across conditions.

Condition-specific learning rates were used:

- Pretrained: `1e-4`
- Scratch: `3e-4`

A post-hoc symmetric learning-rate sensitivity analysis was later performed using the same final training pipeline for both conditions.

### Datasets

- BloodMNIST
- DermaMNIST
- PneumoniaMNIST

### Evaluation

The analysis includes:

- Clean balanced accuracy
- Corrupted balanced accuracy
- Balanced Error (BE)
- Relative Balanced Error (rBE)
- Corruption-family effects
- Corruption-severity effects
- Training-set-size ablation
- Clean-performance / robustness relationships
- Chance-normalized degradation sensitivity analyses

### Experimental scale

The preregistered study included:

- **30 core training runs**
  - 3 datasets
  - 2 initialization conditions
  - 5 random seeds
- **18 reduced-data training runs**
  - BloodMNIST
  - 3 reduced training fractions
  - 2 initialization conditions
  - 3 seeds

Total: **48 trained models**

Additional validation-only runs were later used for a post-hoc learning-rate sensitivity check.

---

## Main findings

The results do **not** support a uniform corruption-robustness benefit from ImageNet initialization.

### H1 — Corruption-family dependence

**Rejected.**

The preregistered hypothesis predicted that the benefit of ImageNet initialization would be smaller for task-specific corruptions than for pooled noise, blur, and digital corruptions.

Instead, the interaction was reversed in direction on both preregistered datasets:

- BloodMNIST: `+1.0009 ΔBE`
- DermaMNIST: `+0.5657 ΔBE`

The bootstrap interval excluded zero on DermaMNIST but narrowly included zero on BloodMNIST.

### H2 — Low-data regime

**Supported.**

On BloodMNIST, the clean balanced-accuracy advantage of ImageNet initialization increased as training data decreased:

- 100% data: `+0.47 pp`
- 50% data: `+0.79 pp`
- 25% data: `+1.03 pp`
- 10% data: `+3.72 pp`

A post-hoc corruption extension showed the largest corrupted-performance advantage in the most data-limited condition:

- 100% data: `+16.58 pp`
- 10% data: `+23.40 pp`

The intermediate corrupted-data fractions were not strictly monotonic.

### H3 — Clean performance versus robustness

**Rejected according to the preregistered analysis.**

Across pooled dataset-seed observations, larger clean-accuracy gains did not reliably correspond to larger corruption-robustness gains. Post-hoc within-dataset and dataset-centered analyses showed that the pooled negative association was strongly influenced by between-dataset differences.

### H4 — Severity dependence

**Rejected.**

Only **2 of 14 dataset-by-corruption-family cells (14.3%)** showed the preregistered strictly monotonic decrease in pretraining advantage as corruption severity increased:

- DermaMNIST blur
- PneumoniaMNIST digital

---

## Additional sensitivity analyses

Several analyses were added during revision and are explicitly treated as post hoc.

### Learning-rate sensitivity

A symmetric validation-only search using the final BloodMNIST pipeline recovered the same learning rates used in the core experiment:

- Pretrained: `1e-4`
- Scratch: `3e-4`

### Clean-performance and floor-effect sensitivity

The heterogeneous corruption results were not explained solely by differences in clean performance or by performance approaching chance level.

### BloodMNIST color-family sensitivity

All five BloodMNIST color corruptions favored ImageNet initialization in ΔBE:

- Range: `+1.29` to `+12.65`
- Median: `+4.33`
- Mean: `+5.53`

After excluding the largest value, the remaining four-corruption mean was still approximately `+3.75 ΔBE`.

---

## Preregistration and deviations

The study was preregistered before the core experiments in a timestamped Git commit:

**Preregistration commit:** [`5b1f1b4`](https://github.com/rifatabrarjowad/medmnist-pretraining-robustness/commit/5b1f1b4)

The repository also contains:

- [`preregistration.md`](preregistration.md)
- [`DEVIATIONS.md`](DEVIATIONS.md)

Documented deviations and revision-stage corrections include:

- correction of PneumoniaMNIST gamma corruptions to the task-specific family;
- clarification of the implemented BloodMNIST reduced-data sampling procedure;
- post-hoc learning-rate sensitivity analysis;
- post-hoc corruption evaluation of the BloodMNIST data-size ablation;
- post-hoc H3 decomposition and degradation-sensitivity analyses.

No preregistered hypothesis was retrospectively redefined based on the observed results.

---

## Repository structure

```text
.
├── configs/            # experiment configuration files
├── experiments/        # experiment and diagnostic runners
├── figures/            # generated manuscript figures
├── notebooks/          # exploratory / supporting notebooks
├── paper/              # manuscript source and final PDF
├── results/            # result tables and statistical outputs
├── src/                # training, evaluation, analysis, and plotting code
├── preregistration.md  # preregistered hypotheses and analysis plan
├── DEVIATIONS.md       # documented deviations and revision-stage analyses
├── README.md
└── LICENSE
```

---

## Manuscript

The current non-blind manuscript is available at:

[`paper/final_manuscript.pdf`](paper/final_manuscript.pdf)

Editable manuscript source:

[`paper/manuscript.md`](paper/manuscript.md)

---

## Reproducibility

The repository contains scripts for model training, clean evaluation, MedMNIST-C corruption evaluation, corruption-family analysis, severity analysis, H2 training-fraction analysis, H3 correlation and dataset-centered analyses, degradation/floor-effect sensitivity analysis, and figure generation.

The main pipeline uses PyTorch, torchvision, MedMNIST, MedMNIST-C, scikit-learn, pandas, NumPy, SciPy, and Matplotlib.

Training was performed using the PyTorch MPS backend on Apple Silicon when available.

Model checkpoints and raw datasets are not committed because of file-size and dataset-distribution considerations.

---

## Important methodological notes

- No stochastic data augmentation was used in the primary experiments.
- Five seeds were used for the core experiment.
- Three seeds were used for the reduced-data BloodMNIST conditions.
- Reduced-data subsets were deterministic and matched between pretrained and scratch conditions, but were not explicitly class-stratified as originally specified in the preregistration.
- Post-hoc analyses are explicitly labeled throughout the manuscript and repository.
- Balanced Error (BE) is treated as the primary normalized corruption metric; rBE is interpreted cautiously where benchmark denominators are small or negative.

---

## Author

**Rifat Abrar Jowad**  
Independent Researcher  
Austin, Texas, USA

Email: [rajdiprifat2004@gmail.com](mailto:rajdiprifat2004@gmail.com)  
GitHub: [@rifatabrarjowad](https://github.com/rifatabrarjowad)

---

## Citation

A formal citation will be added after the manuscript receives a persistent preprint or publication identifier.

Until then:

```text
Jowad, R. A. Inherited or Learned? Decomposing the Contribution of
ImageNet Pretraining to Corruption Robustness in Medical Image Classification.
GitHub repository, 2026.
https://github.com/rifatabrarjowad/medmnist-pretraining-robustness
```

---

## License

See the [`LICENSE`](LICENSE) file for licensing information.
