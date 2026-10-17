# Results

## Clean and corrupted performance

ImageNet initialization improved clean balanced accuracy on all three datasets, but its effect on corruption performance varied substantially across datasets. On BloodMNIST, pretrained models achieved 99.02% mean clean balanced accuracy compared with 98.60% for models trained from scratch. Under corruption, this difference widened substantially: pretrained models achieved 89.01% mean corrupted balanced accuracy, compared with 72.64% for scratch models.

On DermaMNIST, ImageNet initialization produced a much larger clean-set advantage, with mean clean balanced accuracy of 72.65% for pretrained models and 55.38% for scratch models. This advantage did not carry over to corrupted performance. Mean corrupted balanced accuracy was 40.37% for pretrained models and 40.86% for scratch models.

A similar divergence appeared on PneumoniaMNIST. Pretrained models achieved 88.90% mean clean balanced accuracy compared with 80.87% for scratch models, but under corruption the pretrained models performed worse on average, with 72.19% balanced accuracy compared with 76.54% for scratch models.

Thus, improved clean performance from ImageNet initialization did not consistently correspond to improved performance under corruption.

## Corruption-family dependence

The effect of ImageNet initialization varied strongly across corruption families.

On BloodMNIST, color corruptions showed the largest mean pretraining advantage, with a pretrained-minus-scratch balanced-accuracy difference of +30.74 percentage points and a mean ΔBE of +5.53. In a post-hoc descriptive check, all five BloodMNIST color corruptions favored pretraining in ΔBE (range +1.29 to +12.65; median +4.33); after excluding the largest value from saturate, the remaining four-corruption mean was +3.75, indicating that the family-level pattern was not driven solely by a single corruption. Task-specific corruptions also favored pretrained models (+14.04 points; ΔBE = +0.77), as did blur corruptions (+6.46 points; ΔBE = +0.32). Digital corruptions instead favored scratch training (-7.27 points; ΔBE = -0.78).

On DermaMNIST, task-specific corruptions showed the largest mean pretraining benefit (+21.17 percentage points; ΔBE = +0.36), followed by color corruptions (+16.29 points; ΔBE = +0.33). In contrast, blur (-18.22 points; ΔBE = -0.29), digital (-14.85 points; ΔBE = -0.31), and noise (-7.60 points; ΔBE = -0.10) favored scratch-trained models.

PneumoniaMNIST also included task-specific gamma corruptions. These task-specific corruptions favored pretraining (+8.79 percentage points; ΔBE = +0.72), as did color corruptions (+9.97 points; ΔBE = +0.56). In contrast, noise (-21.01 points; ΔBE = -0.50), digital (-10.69 points; ΔBE = -0.41), and blur (-8.57 points; ΔBE = -0.31) favored scratch training.

The preregistered H1 analysis was defined for BloodMNIST and DermaMNIST and predicted that the benefit of ImageNet initialization would be smaller for task-specific corruptions than for pooled noise, blur, and digital corruptions. The observed interaction was reversed in direction on both datasets: +1.0009 ΔBE on BloodMNIST and +0.5657 ΔBE on DermaMNIST. Paired-seed bootstrap 95% confidence intervals were [-0.0094, 2.0111] for BloodMNIST and [0.4799, 0.6604] for DermaMNIST; thus, the interval narrowly included zero on BloodMNIST but excluded zero on DermaMNIST. These bootstrap intervals are reported as descriptive uncertainty estimates rather than multiplicity-adjusted confirmatory interaction tests.

Because the observed interaction was opposite to the preregistered directional prediction, H1 was rejected.

The corresponding PneumoniaMNIST task-specific results are reported descriptively and were not part of the preregistered H1 contrast.

Separately from the preregistered H1 interaction analysis, dataset-level two-sided Wilcoxon signed-rank tests on per-corruption ΔBE values yielded W = 9, p = 0.032 for BloodMNIST; W = 57, p = 0.890 for DermaMNIST; and W = 38, p = 0.635 for PneumoniaMNIST. Thus, the per-corruption ΔBE distribution provided evidence of an overall positive pretraining effect on BloodMNIST, but not on DermaMNIST or PneumoniaMNIST at the nominal 0.05 level. Family-level Wilcoxon tests were additionally adjusted within dataset using the Holm-Bonferroni procedure; none remained significant after correction. Because individual families contained only 1–5 corruption types, these family-level tests had very limited inferential power, and the absence of statistical significance should not be interpreted as evidence of no family-level effect.

## Training-set size ablation

The preregistered H2 analysis examined whether the benefit of ImageNet initialization increased as the available BloodMNIST training data decreased.

For clean balanced accuracy, the pretrained-minus-scratch gap was +0.47 percentage points at 100% of the training set (95% bootstrap CI: -0.19 to +1.29), +0.79 points at 50% (95% CI: +0.69 to +0.96), +1.03 points at 25% (95% CI: +0.90 to +1.17), and +3.72 points at 10% (95% CI: +2.36 to +6.09). The clean-set gap therefore increased monotonically as the available training data decreased.

We additionally evaluated the same training-fraction models under BloodMNIST-C corruptions. Averaged across all 11 corruptions and five severity levels, the pretrained-minus-scratch corrupted balanced-accuracy gap was +16.58 percentage points at 100% of the training data (95% bootstrap CI: +15.94 to +17.14), +18.12 points at 50% (95% CI: +17.12 to +19.39), +17.07 points at 25% (95% CI: +15.27 to +20.49), and +23.40 points at 10% (95% CI: +19.89 to +28.62). Unlike the clean-set gaps, the corrupted-performance gaps were not strictly monotonic across all intermediate fractions, but the largest advantage occurred in the most data-limited 10% condition.

A paired seed-wise comparison of the 10% and 100% conditions showed that reducing the training set from 100% to 10% increased the pretraining advantage by a mean of +3.25 percentage points on clean data and +6.83 points under corruption. All three paired seeds showed an increase in both analyses. Because the reduced-data experiments used only three paired seeds, the associated bootstrap intervals are interpreted descriptively.

Taken together, these results support H2: the advantage of ImageNet initialization was larger in the most data-limited BloodMNIST regime, both on clean data and under corruption.

## Clean accuracy versus corruption robustness

The preregistered H3 analysis examined whether gains under corruption were largely explained by gains on clean data.

At the dataset level, the patterns differed sharply. On BloodMNIST, the mean clean balanced-accuracy gain from pretraining was only +0.42 percentage points, whereas the mean corrupted-set gain was +16.38 points. On DermaMNIST, the clean gain was +17.28 points, but the corrupted-set gain was -0.48 points. On PneumoniaMNIST, the clean gain was +8.03 points, while the corrupted-set gain was -4.35 points.

Across the 15 paired dataset-seed observations, clean-accuracy gain was negatively associated with corrupted-accuracy gain (Pearson r = -0.63; Spearman ρ = -0.46). Clean gain was also negatively associated with BE advantage (Pearson r = -0.79; Spearman ρ = -0.63). Because these observations comprise repeated seeds from only three datasets, these pooled correlations are descriptive rather than independent-sample confirmatory tests.

A post-hoc decomposition showed that the pooled negative association was strongly influenced by between-dataset differences. After centering clean gain and robustness gain within each dataset, the association between clean gain and corrupted-accuracy gain became positive (Pearson r = +0.74; Spearman ρ = +0.69). The centered association between clean gain and BE advantage was also positive (Pearson r = +0.58; Spearman ρ = +0.44).

Within-dataset correlations were heterogeneous and were based on only five paired seeds per dataset. For clean gain versus corrupted-accuracy gain, Pearson correlations were +0.18 on BloodMNIST, +0.76 on DermaMNIST, and +0.81 on PneumoniaMNIST. The corresponding Spearman correlations were -0.20, +0.60, and +0.50. For clean gain versus BE advantage, Pearson correlations were -0.65 on BloodMNIST, +0.82 on DermaMNIST, and +0.84 on PneumoniaMNIST. These within-dataset estimates are therefore treated as descriptive sensitivity analyses rather than confirmatory hypothesis tests.

The preregistered pooled analysis did not support the expectation that gains under corruption were largely explained by clean-set gains, and H3 was rejected according to the preregistered analysis. However, the post-hoc decomposition indicates that the negative pooled association should not be interpreted as evidence that larger clean gains intrinsically reduce corruption robustness.

## Sensitivity to clean-performance and chance-level differences

Because pretrained and scratch models differed substantially in clean accuracy on DermaMNIST and PneumoniaMNIST, we conducted a post-hoc sensitivity analysis to assess whether the corruption results could be explained by clean-performance differences or floor effects.

For each dataset, we measured the absolute drop from clean to mean corrupted balanced accuracy and the fraction of above-chance clean performance retained under corruption. Chance balanced accuracy was defined as 1/8 for BloodMNIST, 1/7 for DermaMNIST, and 1/2 for PneumoniaMNIST.

On BloodMNIST, pretrained models showed a smaller absolute clean-to-corrupted drop than scratch models (10.00 vs. 25.96 percentage points) and retained a larger fraction of their above-chance clean performance (0.884 vs. 0.699).

On DermaMNIST, pretrained models showed a larger absolute drop than scratch models (32.28 vs. 14.52 percentage points) and retained a smaller fraction of above-chance clean performance (0.447 vs. 0.647).

On PneumoniaMNIST, pretrained models likewise showed a larger absolute drop (16.71 vs. 4.33 percentage points) and retained less above-chance clean performance (0.568 vs. 0.861).

Thus, the heterogeneous corruption results were not explained solely by the lower clean accuracy of scratch models or by chance-level floor effects. The sensitivity analysis reinforced the dataset-dependent pattern: pretraining improved retained corruption performance on BloodMNIST but not on DermaMNIST or PneumoniaMNIST.

## Severity dependence

The preregistered H4 predicted that the pretrained-minus-scratch balanced-accuracy gap would decrease monotonically as corruption severity increased from level 1 to level 5.

This pattern was uncommon. Only 2 of 14 dataset-by-corruption-family cells (14.3%) showed a strictly monotonic decrease across severity levels. These were DermaMNIST blur and PneumoniaMNIST digital corruptions.

At the dataset level, BloodMNIST showed the opposite overall tendency. The mean pretraining advantage increased from +4.44 percentage points at severity 1 to +14.90 points at severity 2 and +21.50 points at severity 3, before remaining high at +21.13 points at severity 4 and +19.93 points at severity 5.

DermaMNIST shifted from a small pretrained advantage at lower severity (+1.79 points at severity 1 and +1.78 points at severity 2) to a scratch advantage at higher severity (-0.63 points at severity 3, -2.53 points at severity 4, and -2.84 points at severity 5).

PneumoniaMNIST increasingly favored scratch training as severity increased, from -0.33 percentage points at severity 1 to -2.71 points at severity 2, -4.92 points at severity 3, -6.03 points at severity 4, and -7.76 points at severity 5.

Because strictly monotonic decreases occurred in only a small minority of dataset-by-family cells, H4 was rejected.

## Summary of hypothesis tests

Of the four preregistered hypotheses, only H2 was supported. H1, H3, and H4 were rejected according to their preregistered directional predictions and analysis definitions.

The additional post-hoc analyses refined, but did not replace, these preregistered conclusions. Corruption evaluation of the BloodMNIST data-size ablation showed the largest pretraining advantage in the 10% training regime; dataset-centered H3 analyses showed that the negative pooled clean-versus-robustness association was strongly influenced by between-dataset differences; and chance-normalized degradation analyses showed that the heterogeneous robustness effects were not explained solely by clean-performance differences or floor effects.

Overall, ImageNet initialization did not produce a uniform corruption-robustness benefit. Its effect depended strongly on the dataset, corruption family, training-data availability, and corruption severity.
