# Inherited or Learned? Decomposing the Contribution of ImageNet Pretraining to Corruption Robustness in Medical Image Classification

This repository contains the code, analysis, figures, and manuscript for a controlled study of how much medical-image corruption robustness is attributable to ImageNet initialization.

## Research question

To what extent is corruption robustness on MedMNIST-C inherited from ImageNet pretraining rather than learned from the target medical dataset?

## Experimental design

We compare two ResNet-18 initialization conditions:

- ImageNet-pretrained (`IMAGENET1K_V1`)
- Random initialization (`weights=None`)

All other major experimental choices are held fixed, including architecture, preprocessing, optimization, validation procedure, and evaluation protocol.

Datasets:

- BloodMNIST
- DermaMNIST
- PneumoniaMNIST

Evaluation:

- Clean balanced accuracy
- MedMNIST-C corruption performance
- Balanced Error (BE)
- Relative Balanced Error (rBE)
- Corruption-family effects
- Corruption severity
- Training-set-size ablation

The study includes 30 core runs and 18 training-fraction ablation runs, for 48 training runs in total.

## Main findings

The results do not support a uniform robustness benefit from ImageNet initialization.

- H1 — Rejected: task-specific corruptions did not show the predicted smaller pretraining benefit.
- H2 — Supported: the clean balanced-accuracy advantage of pretraining increased as BloodMNIST training data decreased.
- H3 — Rejected: clean-accuracy gains did not reliably predict corruption-robustness gains.
- H4 — Rejected: the pretraining advantage did not decrease monotonically with corruption severity in most dataset-by-family cells.

Overall, the effect of ImageNet initialization depends strongly on dataset, corruption family, training-data availability, and corruption severity.

## Repository structure

```text
.
├── experiments/        # experiment runners
├── figures/            # generated figures
├── paper/              # manuscript and final PDF
├── results/            # clean, corruption, and statistical result tables
├── src/                # training, evaluation, analysis, and plotting code
└── README.md
```

## Manuscript

The current preprint manuscript is available at:

`paper/final_manuscript.pdf`

## Reproducibility

The repository contains the scripts used for training, corruption evaluation, statistical analysis, and figure generation.

Model checkpoints and raw datasets are not included in the repository because of file size and dataset-distribution considerations.

## Author

Rifat Abrar Jowad
St. Edward’s University, Austin, Texas, USA

## Citation

A formal citation will be added after the preprint receives an arXiv or DOI identifier.

## License

Code and manuscript licensing information will be added before archival release.

```

```
