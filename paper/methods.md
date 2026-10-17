# Methods

## Study design

We conducted a controlled comparison of ImageNet-pretrained and randomly initialized ResNet-18 models for medical image classification under common image corruptions. The architecture, official data splits, input preprocessing, optimizer family, learning-rate schedule, stopping criteria, model-selection rule, and evaluation pipeline were held fixed across initialization conditions. The two conditions differed in model initialization and in the learning rate used for training, as described below.

The study was preregistered before the core experiments in a timestamped Git commit on August 30, 2026 (commit `5b1f1b4`; https://github.com/rifatabrarjowad/medmnist-pretraining-robustness/commit/5b1f1b4). The primary endpoint was the interaction between initialization condition and corruption family, expressed using the difference in balanced error (BE) between pretrained and scratch models. Secondary preregistered analyses examined clean and corrupted balanced accuracy, training-set size, corruption severity, and the relationship between clean-accuracy gain and robustness gain.

Additional analyses introduced during revision are explicitly identified as post-hoc sensitivity or descriptive analyses and were not used to redefine the preregistered hypotheses.

## Datasets

We evaluated three datasets from the MedMNIST+ collection at 224 × 224 resolution [4]:

- BloodMNIST: eight-class blood-cell microscopy classification, with 11,959 training, 1,712 validation, and 3,421 test examples.
- DermaMNIST: seven-class dermatoscopic lesion classification, with 7,007 training, 1,003 validation, and 2,005 test examples.
- PneumoniaMNIST: binary chest X-ray classification, with 4,708 training, 524 validation, and 624 test examples.

Official predefined train, validation, and test splits were used without re-splitting.

Corrupted evaluation sets were obtained from MedMNIST-C [5]. BloodMNIST-C contained 11 corruption types, DermaMNIST-C contained 15, and PneumoniaMNIST-C contained 13. Each corruption was evaluated at five severity levels.

BloodMNIST-C included the task-specific corruptions `stain_deposit` and `bubble`. DermaMNIST-C included the task-specific corruptions `black_corner` and `characters`. PneumoniaMNIST-C included the task-specific gamma corruptions `gamma_corr_up` and `gamma_corr_down`.

For the preregistered H1 contrast, task-specific-versus-generic corruption-family comparisons were defined for BloodMNIST and DermaMNIST. PneumoniaMNIST task-specific results were analyzed descriptively during revision and were not added retrospectively to the preregistered H1 test.

## Model

All primary experiments used ResNet-18.

For the pretrained condition, the network was initialized with the torchvision `IMAGENET1K_V1` ResNet-18 weights. For the scratch condition, the same architecture was initialized randomly using `weights=None`.

The final fully connected layer was replaced to match the number of classes in each dataset.

No alternative architecture was used in the primary experiment.

## Input preprocessing

All images were loaded at 224 × 224 resolution and converted to three-channel RGB input using the MedMNIST `as_rgb=True` option.

Both initialization conditions used the same ImageNet normalization:

- mean = [0.485, 0.456, 0.406]
- standard deviation = [0.229, 0.224, 0.225]

No stochastic data augmentation was used during training. Inputs were converted to tensors and normalized using ImageNet mean and standard deviation; no random crop, flip, rotation, or corruption-based augmentation was applied.

## Training procedure

Models were trained using AdamW with a batch size of 64.

The pretrained condition used a learning rate of 1 × 10^-4. The scratch condition used a learning rate of 3 × 10^-4.

A cosine annealing learning-rate schedule was used with a maximum training budget of 50 epochs. Early stopping was based on validation balanced accuracy with a patience of 10 epochs. The checkpoint corresponding to the highest validation balanced accuracy was retained for evaluation.

## Learning-rate selection and sensitivity check

Before the core experiment, the scratch learning rate was selected using a validation-based search on BloodMNIST seed 0 over:

- 1 × 10^-4
- 3 × 10^-4
- 1 × 10^-3

That original search evaluated scratch initialization only and used a shorter training-search configuration than the final core protocol. It also converted BloodMNIST input to three channels through a grayscale-to-RGB transform rather than using the final `as_rgb=True` input pipeline. The best validation balanced accuracy in that search was obtained at 3 × 10^-4, which was then fixed for all scratch models in the core experiment. The pretrained learning rate of 1 × 10^-4 was fixed for the pretrained condition.

Because this original tuning procedure was asymmetric across initialization conditions, we conducted a post-hoc symmetric learning-rate sensitivity check during revision. Both pretrained and scratch models were evaluated on BloodMNIST seed 0 using the same candidate grid of 1 × 10^-4, 3 × 10^-4, and 1 × 10^-3, the final `as_rgb=True` preprocessing pipeline, the same AdamW optimizer and cosine schedule, a 50-epoch maximum, and patience of 10 epochs. Selection was based only on validation balanced accuracy; test-set performance was not used.

The validation-optimal rates from this symmetric check matched the rates used in the core experiment: 1 × 10^-4 for pretrained models and 3 × 10^-4 for scratch models. This sensitivity analysis was post hoc and does not replace the original learning-rate selection procedure.

## Random seeds and core experiment

The core experiment used five random seeds:

0, 1, 2, 3, and 4.

For each of the three datasets, both initialization conditions were trained for all five seeds:

3 datasets × 2 initialization conditions × 5 seeds = 30 core training runs.

No completed seed was removed as an outlier.

## Training-set size ablation

To test the effect of data availability, an additional BloodMNIST ablation was conducted using 50%, 25%, and 10% of the original training set.

For each reduced fraction, pretrained and scratch models were trained using seeds 0, 1, and 2. The 100% condition used the corresponding core experiment models for seeds 0–2.

The reduced training subsets were sampled deterministically using a seed-specific random permutation of the BloodMNIST training indices. The first fraction-specific prefix of that permutation was used, making the 10%, 25%, and 50% subsets nested within seed. The same subset was therefore used for pretrained and scratch conditions at a given fraction and seed. Although the preregistration specified class-stratified subsampling, the implemented procedure was not explicitly stratified; this deviation was identified during revision and is documented in the accompanying deviation log. Validation and test sets were unchanged.

The reduced-data ablation therefore contributed:

3 reduced fractions × 2 initialization conditions × 3 seeds = 18 additional training runs.

Together, the preregistered core experiment and reduced-data ablation comprised 48 trained models. The six validation-only runs used for the later symmetric learning-rate sensitivity check were separate post-hoc diagnostic runs.

The preregistered H2 analysis evaluated clean balanced accuracy across the four training fractions. During revision, the same trained models were additionally evaluated on all BloodMNIST-C corruptions and severity levels to determine whether the low-data pattern also appeared under corruption. This corruption analysis of the training-fraction ablation was post hoc.

## Clean-set evaluation

Each selected checkpoint was evaluated on the official clean test split.

The primary clean-set metric was balanced accuracy, which assigns equal weight to each class and is therefore less sensitive than ordinary accuracy to class imbalance.

Results were summarized across matched random seeds.

## Corruption evaluation

For each dataset, every trained core checkpoint was evaluated on all available MedMNIST-C corruptions at severity levels 1 through 5.

Balanced accuracy was retained separately for each corruption type, severity level, initialization condition, and seed.

Corruptions were grouped into the following families where applicable:

- digital
- noise
- blur
- color/intensity
- task-specific

The effect of pretraining was oriented so that positive values indicated an advantage for ImageNet initialization.

For balanced accuracy:

ΔbACC = bACC_pretrained − bACC_scratch.

For balanced error:

ΔBE = BE_scratch − BE_pretrained.

Thus, positive ΔbACC or ΔBE indicates an advantage for the pretrained condition.

## Balanced error and relative balanced error

We used the MedMNIST-C evaluation framework for normalized robustness metrics.

Balanced error (BE) was treated as the primary normalized corruption metric. Relative balanced error (rBE) was treated as a secondary metric.

The implemented MedMNIST-C AlexNet baseline values were used for normalization [5].

For corruption c:

BE_c =
mean model corrupted balanced error for corruption c
/
AlexNet balanced error for corruption c.

Relative balanced error was defined as:

rBE_c =
(model corrupted error − model clean error)
/
(AlexNet corrupted error − AlexNet clean error).

BE values were first computed at the corruption level after aggregating across severity levels according to the evaluation pipeline.

## rBE denominator handling

A known limitation of rBE occurs when the AlexNet denominator is very small or negative.

For PneumoniaMNIST, `contrast_up` and `gamma_corr_up` produced negative AlexNet rBE denominators and were excluded from rBE aggregates according to the preregistered denominator rule.

Corruptions with denominator magnitude below 0.05 were flagged as potentially unstable. These flags were retained in the result tables so that rBE values could be interpreted cautiously.

No corresponding exclusion was applied to BE, which remained the primary normalized corruption metric.

Absolute degradation from clean to corrupted balanced accuracy was also examined alongside normalized metrics.

## Primary H1 analysis

The preregistered primary endpoint tested whether the benefit of ImageNet initialization was smaller for task-specific corruptions than for pooled noise, blur, and digital corruptions on BloodMNIST and DermaMNIST.

For each corruption, performance was aggregated across severity levels, and pretrained-versus-scratch effects were paired using matched experimental structure.

The primary interaction was defined as:

mean ΔBE_task-specific
−
mean ΔBE_noise/blur/digital.

Under this definition, a positive interaction indicates a larger pretraining advantage for task-specific corruptions. The preregistered directional hypothesis predicted the opposite pattern.

PneumoniaMNIST task-specific gamma corruptions were not included retrospectively in this confirmatory H1 contrast.

## Statistical analysis

Main performance summaries were reported as means and standard deviations across seeds where applicable.

For corruption-level robustness comparisons, corruption type was treated as the unit of pairing.

Two-sided Wilcoxon signed-rank tests were applied to per-corruption ΔBE values within each dataset.

Bootstrap 95% confidence intervals were estimated using 10,000 resamples for family-level effects and for the primary task-specific interaction. Because some corruption-family comparisons contained very few corruption types, these bootstrap intervals were treated primarily as descriptive uncertainty estimates.

Holm-Bonferroni correction was applied across the family-level Wilcoxon tests within each dataset. The nominal significance threshold was α = 0.05.

For the training-fraction and severity hypotheses, direction, effect size, paired-seed behavior, and uncertainty intervals were emphasized rather than fitting additional regression models.

## H2 low-data analysis

H2 tested whether the balanced-accuracy advantage of pretraining increased as the available BloodMNIST training data decreased from 100% to 50%, 25%, and 10%.

For clean performance, the pretrained-minus-scratch balanced-accuracy gap was calculated separately at each training fraction.

The post-hoc corruption extension evaluated the same models across all 11 BloodMNIST-C corruption types and five severity levels. For each condition and seed, corrupted balanced accuracy was averaged across corruption types and severity levels, after which pretrained-minus-scratch gaps were calculated.

To directly compare the most data-rich and most data-limited regimes, we also computed the paired seed-wise change in the pretraining advantage between the 100% and 10% training fractions. This comparison was performed separately for clean and corrupted balanced accuracy.

Because only three paired seeds were available for the reduced-data conditions, bootstrap intervals from these analyses were interpreted descriptively.

## H3 clean-performance versus robustness analysis

H3 tested whether gains in corrupted performance were largely explained by clean-set gains.

For each dataset and seed, pretrained-minus-scratch differences were computed for:

- clean balanced accuracy,
- mean corrupted balanced accuracy,
- absolute clean-to-corrupted degradation,
- BE,
- rBE.

The preregistered analysis examined the relationship between clean-accuracy gain and robustness gain across the 15 paired dataset-seed observations from three datasets and five seeds.

Because these observations contain repeated seeds nested within only three datasets, the pooled Pearson and Spearman correlations were treated descriptively rather than as independent-sample confirmatory inference.

During revision, two post-hoc sensitivity analyses were added to examine whether the pooled association was driven by between-dataset differences.

First, Pearson and Spearman correlations were calculated separately within each dataset using the five paired seeds.

Second, clean gain, corrupted-performance gain, and BE advantage were centered by subtracting their dataset-specific means before calculating pooled correlations on the centered values.

These post-hoc analyses were used to diagnose between-dataset structure and did not alter the preregistered H3 decision rule.

## Clean-performance and floor-effect sensitivity analysis

A post-hoc sensitivity analysis examined whether differences in corruption performance could be explained by different clean-performance baselines or by performance approaching chance level.

For each dataset, initialization condition, and seed, we calculated:

absolute drop =
clean balanced accuracy
−
mean corrupted balanced accuracy.

We also calculated the fraction of above-chance clean performance retained under corruption:

retained fraction =
(mean corrupted balanced accuracy − chance balanced accuracy)
/
(clean balanced accuracy − chance balanced accuracy).

Chance balanced accuracy was defined as:

- BloodMNIST: 1/8
- DermaMNIST: 1/7
- PneumoniaMNIST: 1/2

A retained fraction of 1 indicates complete preservation of above-chance clean performance under corruption, whereas a value of 0 indicates reduction to chance-level balanced accuracy.

The corresponding chance-normalized drop was defined as:

chance-normalized drop = 1 − retained fraction.

This analysis was post hoc and was used only as a robustness sensitivity check.

## H4 severity analysis

H4 tested whether the pretrained-minus-scratch balanced-accuracy gap decreased monotonically as corruption severity increased from level 1 through level 5.

For each dataset × corruption-family cell, pretrained-minus-scratch balanced-accuracy differences were calculated separately at each severity level.

A cell satisfied the preregistered H4 directional pattern only if the gap decreased strictly and monotonically across all five severity levels.

After correcting the PneumoniaMNIST gamma corruptions to the task-specific family, the severity analysis contained 14 dataset-by-family cells in total.

Severity-specific paired-seed bootstrap confidence intervals were additionally calculated as descriptive uncertainty estimates.

## Software and hardware

Experiments were implemented in Python using PyTorch, torchvision, MedMNIST, MedMNIST-C, scikit-learn, pandas, NumPy, SciPy, and Matplotlib.

Training and evaluation were executed using the PyTorch MPS backend on Apple Silicon when available.

All experiment scripts, analysis code, configuration decisions, and result tables are maintained in the accompanying repository for reproducibility.
