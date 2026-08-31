# Results

## Clean and corrupted performance

ImageNet initialization improved clean balanced accuracy on all three datasets, but the effect on corruption performance varied substantially across datasets. On BloodMNIST, the pretrained model achieved 99.02% clean balanced accuracy compared with 98.60% for training from scratch. Under corruption, the gap widened substantially: pretrained models achieved 89.01% mean corrupted balanced accuracy, compared with 72.64% for scratch models.

On DermaMNIST, ImageNet initialization produced a much larger clean-set advantage, with mean clean balanced accuracy of 72.66% for pretrained models and 55.38% for scratch models. However, this advantage did not transfer to corrupted performance. Mean corrupted balanced accuracy was 40.37% for pretrained models and 40.86% for scratch models.

A similar divergence appeared on PneumoniaMNIST. Pretrained models achieved 88.90% clean balanced accuracy compared with 80.87% for scratch models, but under corruption the pretrained models performed worse on average, with 72.19% balanced accuracy compared with 76.54% for scratch models.

These results indicate that improved clean performance from ImageNet initialization did not consistently imply improved corruption robustness.

## Corruption-family dependence

The effect of ImageNet initialization depended strongly on corruption family.

On BloodMNIST, pretraining substantially improved robustness to color corruptions, with a mean balanced-accuracy advantage of +30.74 percentage points and a mean ΔBE of +5.53. Task-specific corruptions also favored pretrained models, with a mean balanced-accuracy advantage of +14.04 points and ΔBE of +0.77. Blur corruptions showed a smaller positive effect (+6.46 points), whereas digital corruptions favored scratch training (-7.27 points).

On DermaMNIST, the pattern was similarly heterogeneous. Task-specific corruptions showed the largest pretraining benefit (+21.17 percentage points), followed by color corruptions (+16.29 points). In contrast, blur (-18.22 points), digital (-14.85 points), and noise (-7.60 points) favored scratch-trained models.

PneumoniaMNIST contained no task-specific corruptions. Color corruptions favored pretraining (+9.58 percentage points), whereas noise (-21.01 points), digital (-10.69 points), and blur (-8.57 points) favored scratch training.

The preregistered H1 predicted that the benefit of ImageNet initialization would be smaller for task-specific corruptions than for pooled noise, blur, and digital corruptions. The observed direction was reversed. The task-specific versus pooled interaction was +1.0009 ΔBE on BloodMNIST and +0.5657 on DermaMNIST. The bootstrap 95% confidence interval for the interaction included zero on BloodMNIST [-0.0094, 2.0111] but excluded zero on DermaMNIST [0.4799, 0.6604]. Therefore, H1 was rejected.

## Training-set size ablation

The benefit of ImageNet initialization increased as the available BloodMNIST training data decreased.

At 100% of the training set, the pretrained-minus-scratch balanced-accuracy gap was +0.47 percentage points (95% CI: -0.19 to +1.29). At 50% of the training set, the gap increased to +0.79 points (95% CI: +0.69 to +0.96). At 25%, it increased further to +1.03 points (95% CI: +0.90 to +1.17). At 10% of the training set, the gap increased sharply to +3.72 points (95% CI: +2.36 to +6.09).

The increase was monotonic as the training fraction decreased, and the confidence intervals for the 10% and 100% settings did not overlap. Therefore, H2 was supported.

## Clean accuracy versus corruption robustness

The relationship between clean accuracy gain and corruption robustness gain did not follow the preregistered expectation.

On BloodMNIST, the mean clean balanced-accuracy gain from pretraining was only +0.42 percentage points, whereas the mean corrupted-set gain was +16.38 points. On DermaMNIST, the clean gain was +17.28 points, but the corrupted-set gain was -0.48 points. On PneumoniaMNIST, the clean gain was +8.03 points, while the corrupted-set gain was -4.35 points.

Across the 15 dataset-seed pairs, clean-accuracy gain was negatively associated with corrupted-accuracy gain (Pearson r = -0.63; Spearman ρ = -0.46). Clean gain was also negatively associated with BE advantage (Pearson r = -0.79; Spearman ρ = -0.63).

Thus, the corrupted-set advantage of ImageNet initialization was not explained by the corresponding clean-set advantage. H3 was rejected.

## Severity dependence

The preregistered H4 predicted that the pretrained-minus-scratch balanced-accuracy gap would decrease monotonically as corruption severity increased.

This pattern was uncommon. Only 2 of 13 dataset × corruption-family cells (15.4%) showed a strictly monotonic decrease across severity levels 1 through 5. These were DermaMNIST blur and PneumoniaMNIST digital corruptions.

At the dataset level, BloodMNIST showed the opposite tendency: the overall pretraining advantage increased from +4.44 percentage points at severity 1 to +19.93 points at severity 5, with the largest gap occurring at severity 3. DermaMNIST shifted from a small pretrained advantage at severity 1 (+1.79 points) to a scratch advantage at severity 5 (-2.84 points). PneumoniaMNIST also increasingly favored scratch training as severity increased, from -0.33 points at severity 1 to -7.76 points at severity 5.

Because monotonic decreases occurred in only a small minority of dataset × family cells, H4 was rejected.

## Summary of hypothesis tests

Of the four preregistered hypotheses, only H2 was supported. H1, H3, and H4 were rejected. Overall, the results show that ImageNet initialization does not produce a uniform corruption-robustness benefit in medical image classification. Its effect depends strongly on dataset, corruption family, training-set size, and corruption severity.
