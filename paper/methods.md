# Methods

## Study design

We conducted a controlled comparison of ImageNet-pretrained and randomly initialized ResNet-18 models for medical image classification under common image corruptions. The architecture, data splits, preprocessing, optimization procedure, stopping criteria, and evaluation pipeline were held fixed between initialization conditions. The only intended difference between the two primary conditions was model initialization.

The study was preregistered before the core experiments. The primary endpoint was the interaction between initialization and corruption family, expressed using the difference in balanced error (BE) between pretrained and scratch models. Secondary analyses examined clean and corrupted balanced accuracy, training-set size, corruption severity, and the relationship between clean-accuracy gain and robustness gain.

## Datasets

We evaluated three datasets from the MedMNIST+ collection at 224 × 224 resolution:

- BloodMNIST: eight-class blood-cell microscopy classification, with 11,959 training, 1,712 validation, and 3,421 test examples.
- DermaMNIST: seven-class dermatoscopic lesion classification, with 7,007 training, 1,003 validation, and 2,005 test examples.
- PneumoniaMNIST: binary chest X-ray classification, with 4,708 training, 524 validation, and 624 test examples.

Official predefined train, validation, and test splits were used without re-splitting.

Corrupted evaluation sets were obtained from MedMNIST-C. BloodMNIST-C contained 11 corruption types, DermaMNIST-C contained 15, and PneumoniaMNIST-C contained 13. Each corruption was evaluated at five severity levels.

BloodMNIST-C included the task-specific corruptions `stain_deposit` and `bubble`. DermaMNIST-C included `black_corner` and `characters`. PneumoniaMNIST-C contained no task-specific corruption family and was included as a contrast dataset.

## Model

All experiments used ResNet-18.

For the pretrained condition, the network was initialized with the torchvision `IMAGENET1K_V1` ResNet-18 weights. For the scratch condition, the same ResNet-18 architecture was initialized randomly using `weights=None`.

The final fully connected layer was replaced to match the number of classes in each dataset.

No alternative architecture was used in the primary experiment.

## Input preprocessing

All images were loaded at 224 × 224 resolution and converted to three-channel RGB input using the MedMNIST `as_rgb=True` option.

Both pretrained and scratch models used the same ImageNet normalization:

- mean = [0.485, 0.456, 0.406]
- standard deviation = [0.229, 0.224, 0.225]

Using identical normalization in both conditions avoided introducing a preprocessing difference between the pretrained and scratch models.

No corruption-based data augmentation was used during training.

## Training procedure

Models were trained using AdamW with a batch size of 64.

The pretrained condition used a learning rate of 1 × 10^-4.

The scratch learning rate was selected before the main experiment using a small validation-based search on BloodMNIST seed 0 over:

- 1 × 10^-4
- 3 × 10^-4
- 1 × 10^-3

The selected scratch learning rate was 3 × 10^-4 and was then fixed for all subsequent datasets and seeds.

A cosine annealing learning-rate schedule was used with a maximum training budget of 50 epochs.

Early stopping was based on validation balanced accuracy with patience of 10 epochs. The checkpoint corresponding to the highest validation balanced accuracy was retained for evaluation.

## Random seeds and core experiment

The core experiment used five random seeds:

0, 1, 2, 3, and 4.

For each of the three datasets, both initialization conditions were trained for all five seeds:

3 datasets × 2 initialization conditions × 5 seeds = 30 core training runs.

No completed seed was removed as an outlier.

## Training-set size ablation

To test the effect of data availability, an additional BloodMNIST ablation was conducted using 50%, 25%, and 10% of the original training set.

For each fraction, pretrained and scratch models were trained using seeds 0, 1, and 2.

The 100% setting used the corresponding core experiment models for seeds 0–2.

The reduced training subsets were sampled deterministically using the experiment seed. Validation and test sets were unchanged.

The ablation therefore contributed:

3 reduced fractions × 2 initialization conditions × 3 seeds = 18 additional runs.

Across the core and ablation experiments, 48 training runs were completed.

## Clean-set evaluation

Each selected checkpoint was evaluated on the official clean test split.

The primary clean-set metric was balanced accuracy, which gives equal weight to each class and is therefore less sensitive than ordinary accuracy to class imbalance.

Results were summarized as mean ± standard deviation across seeds.

## Corruption evaluation

For each dataset, every trained core checkpoint was evaluated on all available MedMNIST-C corruptions at severity levels 1 through 5.

Balanced accuracy was retained separately for every corruption and severity level.

Corruptions were also grouped into families including digital, noise, blur, color/intensity, and task-specific corruptions where applicable.

The effect of pretraining was expressed so that positive values indicated an advantage for ImageNet initialization.

For balanced accuracy:

ΔbACC = bACC_pretrained − bACC_scratch.

For balanced error:

ΔBE = BE_scratch − BE_pretrained.

Thus, positive ΔBE indicates lower normalized balanced error for the pretrained condition.

## Balanced error and relative balanced error

We used the MedMNIST-C evaluation framework for normalized robustness metrics.

Balanced error (BE) was treated as the primary normalized robustness metric.

Relative balanced error (rBE) was treated as a secondary metric.

The implemented MedMNIST-C AlexNet baseline values were used for normalization.

For corruption c:

BE_c = mean corrupted balanced error / AlexNet balanced error for corruption c.

Relative balanced error adjusts the corrupted error relative to clean error:

rBE_c =
(model corrupted error − model clean error)
/
(AlexNet corrupted error − AlexNet clean error).

## rBE denominator handling

A known issue occurs when the denominator of rBE is very small or negative.

For PneumoniaMNIST, `contrast_up` and `gamma_corr_up` produced negative AlexNet rBE denominators. These corruptions were therefore excluded from rBE aggregates according to the preregistered rule.

Corruptions with denominator magnitude below 0.05 were flagged as potentially unstable.

No such exclusion was applied to BE, which remained the primary normalized metric.

Absolute balanced-accuracy degradation from clean to corrupted performance was also reported alongside normalized metrics.

## Primary analysis

The preregistered primary endpoint tested whether the effect of ImageNet initialization differed between task-specific corruptions and pooled noise, blur, and digital corruptions on BloodMNIST and DermaMNIST.

For each corruption, performance was first averaged across severity levels and seeds.

The primary interaction was defined as:

mean ΔBE_task-specific − mean ΔBE_noise/blur/digital.

A positive interaction indicates a larger pretraining advantage for task-specific corruptions.

## Statistical analysis

All main performance results were summarized as mean ± standard deviation over seeds.

For corruption-level robustness comparisons, the corruption type was the unit of pairing.

Two-sided Wilcoxon signed-rank tests were applied to per-corruption ΔBE values within each dataset.

Bootstrap 95% confidence intervals were estimated using 10,000 resamples for family-level ΔBE and for the primary task-specific interaction.

Holm–Bonferroni correction was applied across family-level tests within each dataset.

The significance threshold was α = 0.05.

For the training-fraction and severity hypotheses, direction, effect size, and confidence intervals were emphasized rather than regression p-values.

## Hypothesis-specific analyses

H1 tested whether the ImageNet pretraining benefit was smaller for task-specific corruptions than for pooled noise, blur, and digital corruptions.

H2 tested whether the balanced-accuracy advantage of pretraining increased as the available BloodMNIST training data decreased from 100% to 50%, 25%, and 10%.

H3 tested whether gains in corrupted performance were largely explained by clean-set gains. We compared paired clean and corrupted balanced-accuracy differences and examined the relationship between clean-accuracy gain and robustness advantage.

H4 tested whether the pretrained-minus-scratch balanced-accuracy gap decreased monotonically as corruption severity increased from level 1 to level 5 across dataset × corruption-family cells.

## Software and hardware

Experiments were implemented in Python using PyTorch, torchvision, MedMNIST, MedMNIST-C, scikit-learn, pandas, NumPy, SciPy, and Matplotlib.

Training and evaluation were executed using the PyTorch MPS backend on Apple Silicon when available.

All experiment scripts, analysis code, configuration decisions, and result tables are maintained in the accompanying public repository for reproducibility.
