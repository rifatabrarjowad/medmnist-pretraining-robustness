# Inherited or Learned? Decomposing the Contribution of ImageNet Pretraining to Corruption Robustness in Medical Image Classification

## Abstract

ImageNet pretraining is widely used in medical image classification, but it remains unclear whether robustness to image corruption is inherited from pretraining or learned primarily from the target medical dataset. We study this question using a controlled comparison between ImageNet-pretrained and randomly initialized ResNet-18 models on BloodMNIST, DermaMNIST, and PneumoniaMNIST, evaluated with MedMNIST-C. Architecture, preprocessing, optimization, model selection, and evaluation procedures were held fixed across initialization conditions.

Across 30 core training runs and 18 training-set-size ablation runs, we evaluated clean balanced accuracy, corruption performance, balanced error, relative balanced error, corruption-family effects, and severity dependence. ImageNet initialization produced strongly heterogeneous robustness effects. On BloodMNIST, pretrained models achieved substantially higher mean corrupted balanced accuracy than scratch models (89.01% vs. 72.64%), whereas on DermaMNIST the two conditions were nearly equal (40.37% vs. 40.86%) and on PneumoniaMNIST scratch models performed better under corruption (76.54% vs. 72.19%).

Contrary to our preregistered hypothesis, task-specific corruptions showed larger rather than smaller pretraining benefits relative to pooled noise, blur, and digital corruptions, particularly on DermaMNIST. Clean-accuracy gains also failed to predict robustness gains: datasets with large clean improvements did not necessarily improve under corruption. In contrast, the benefit of ImageNet initialization increased monotonically as BloodMNIST training data were reduced, from a +0.47 percentage-point balanced-accuracy gap at 100% of the training data to +3.72 points at 10%. Pretraining advantage also did not decrease consistently with corruption severity, with only 2 of 13 dataset-by-family cells exhibiting the preregistered monotonic trend.

These results show that ImageNet initialization does not confer uniform corruption robustness in medical image classification. Its effect depends strongly on corruption family, dataset, training-data availability, and severity, suggesting that clean accuracy and the presence of pretraining alone are insufficient indicators of robustness under distribution shift.

## Introduction

Deep neural networks for medical image classification are commonly initialized from models pretrained on large natural-image datasets such as ImageNet. Transfer learning is attractive because medical datasets are often smaller than general-purpose computer-vision datasets, and pretrained representations can improve optimization and data efficiency. However, the value of ImageNet pretraining in medical imaging is not uniform. Prior work has shown that transfer from natural images can provide only limited benefits on some medical imaging tasks, raising questions about which aspects of model performance are actually inherited from pretraining [3].

A separate concern is robustness to distribution shift. Standard test accuracy measures performance on images drawn from approximately the same distribution as the training data, but deployed medical images may differ because of acquisition artifacts, noise, compression, changes in contrast, blur, staining variation, or other image degradations. Robustness benchmarks based on controlled corruptions provide a reproducible way to study this problem.

The ImageNet-C benchmark established a widely used framework for evaluating model robustness to common image corruptions at multiple severity levels [1]. Subsequent work showed that pretraining can improve robustness and uncertainty in natural-image settings, suggesting that pretrained representations may contribute properties beyond conventional clean-set accuracy [2].

Whether the same relationship holds in medical imaging is less clear. Medical images differ substantially from ImageNet images in texture, color statistics, semantic structure, acquisition process, and task definition. A pretrained model may therefore inherit useful invariances from ImageNet, but it may also inherit sensitivities that are poorly matched to medical image corruptions.

MedMNIST provides a standardized collection of biomedical image-classification datasets spanning multiple modalities and tasks [4]. MedMNIST-C extends this benchmark by introducing generic as well as task- and modality-specific corruptions at multiple severity levels, enabling controlled robustness evaluation across medical imaging datasets [5].

Existing MedMNIST-C evaluations primarily establish corruption benchmarks and robustness-enhancing augmentation strategies. However, corruption performance can reflect several sources simultaneously: architecture, training data, optimization, augmentation, and initialization. This makes it difficult to determine how much robustness is specifically inherited from ImageNet pretraining.

In this work, we isolate the effect of initialization through a controlled comparison of ImageNet-pretrained and randomly initialized ResNet-18 models. Architecture, preprocessing, optimization, model-selection procedure, and evaluation protocol are held fixed, while initialization is varied.

We evaluate BloodMNIST, DermaMNIST, and PneumoniaMNIST using their corresponding MedMNIST-C corruption benchmarks. These datasets provide complementary conditions: BloodMNIST and DermaMNIST contain task-specific corruptions, whereas PneumoniaMNIST provides a contrast setting without task-specific corruptions.

Our analysis addresses four questions. First, does the contribution of ImageNet initialization depend on corruption family? Second, does the benefit of pretraining increase when medical training data are limited? Third, are improvements under corruption largely explained by improvements on clean data? Fourth, does the effect of pretraining change systematically with corruption severity?

The main finding is that ImageNet initialization does not confer a uniform corruption-robustness advantage. Its effect depends strongly on corruption family, dataset, training-set size, and severity. In particular, task-specific corruptions often benefited from pretraining more than generic corruptions, clean-accuracy gains did not predict corruption-robustness gains, and the value of pretraining increased clearly in the low-data BloodMNIST regime.

These results suggest that medical image robustness should not be inferred from clean performance or from the presence of ImageNet pretraining alone. Instead, the contribution of pretraining should be evaluated under the specific distribution shifts that matter for the target imaging setting.

## Related Work

### Corruption robustness benchmarks

Controlled corruption benchmarks provide a standardized framework for measuring performance under non-adversarial image degradation. Hendrycks and Dietterich introduced ImageNet-C and related benchmarks to evaluate classifiers under common corruptions and perturbations rather than worst-case adversarial attacks [1]. ImageNet-C applies multiple corruption types at several severity levels and has become a common reference for evaluating distribution-shift robustness.

The same general framework has increasingly been applied to medical imaging, where acquisition artifacts and modality-specific shifts can differ substantially from those encountered in natural images.

### MedMNIST and MedMNIST-C

MedMNIST v2 provides a standardized collection of lightweight biomedical image-classification datasets spanning multiple medical imaging modalities, dataset sizes, and prediction tasks [4]. The benchmark facilitates reproducible comparisons using common APIs and predefined data splits.

MedMNIST-C extends this collection specifically for corruption robustness [5]. Di Salvo, Doerrich, and Ledig introduced corruptions designed to simulate both generic image degradation and artifacts motivated by particular medical datasets and imaging modalities. The benchmark covers 12 two-dimensional MedMNIST datasets and nine imaging modalities and evaluates corruptions at five severity levels.

Their work demonstrated substantial vulnerability of medical classifiers to corruption and showed that targeted, domain-informed augmentation can improve robustness. Our study uses the MedMNIST-C evaluation framework but asks a different question: rather than proposing a robustness intervention, we isolate how model initialization itself affects corruption performance.

### ImageNet pretraining and robustness

ImageNet pretraining is widely used for transfer learning. Beyond data efficiency and optimization, pretraining has also been associated with robustness-related benefits. Hendrycks, Lee, and Mazeika showed that pretraining can improve robustness and uncertainty across several settings, including label corruption, adversarial examples, out-of-distribution detection, and calibration [2].

These findings motivate the possibility that corruption robustness may be partially inherited from the source pretraining task. However, such conclusions cannot automatically be transferred to medical imaging because of the domain gap between natural and medical images.

Our study therefore does not test whether pretraining can ever improve robustness; that phenomenon has already been established in broader computer-vision research. Instead, we examine how much of MedMNIST-C robustness is attributable specifically to ImageNet initialization and whether that contribution changes across medical datasets, corruption families, data availability, and severity.

### Transfer learning in medical imaging

The usefulness of natural-image pretraining for medical imaging has been debated. Raghu et al. studied transfer learning on medical imaging tasks and found that ImageNet transfer sometimes provided surprisingly limited performance benefits [3]. Their analysis emphasized that the gains associated with transfer may not always arise from sophisticated reuse of high-level ImageNet features.

This work highlights an important distinction between conventional predictive performance and other properties inherited through initialization. Even when clean-set gains are small, pretraining could still affect robustness; conversely, large clean-set improvements might not guarantee robust performance under image degradation.

Our experiments directly examine this distinction by measuring clean accuracy and corruption robustness separately.

### Position of the present study

The present study sits at the intersection of medical transfer learning and corruption robustness.

Rather than comparing different robustness algorithms, architectures, or augmentation methods, we hold these factors fixed and perform a controlled pretrained-versus-scratch comparison. This allows us to decompose the contribution of ImageNet initialization itself.

Our focus is therefore not the broad claim that pretraining improves robustness, but a narrower question: under which medical image corruptions does ImageNet initialization help, under which does it hurt, and how does that effect depend on data availability and corruption severity?

## Methods

### Study design

We conducted a controlled comparison of ImageNet-pretrained and randomly initialized ResNet-18 models for medical image classification under common image corruptions. The architecture, data splits, preprocessing, optimization procedure, stopping criteria, and evaluation pipeline were held fixed between initialization conditions. The only intended difference between the two primary conditions was model initialization.

The study was preregistered before the core experiments. The primary endpoint was the interaction between initialization and corruption family, expressed using the difference in balanced error (BE) between pretrained and scratch models. Secondary analyses examined clean and corrupted balanced accuracy, training-set size, corruption severity, and the relationship between clean-accuracy gain and robustness gain.

### Datasets

We evaluated three datasets from the MedMNIST+ collection at 224 × 224 resolution:

- BloodMNIST: eight-class blood-cell microscopy classification, with 11,959 training, 1,712 validation, and 3,421 test examples.
- DermaMNIST: seven-class dermatoscopic lesion classification, with 7,007 training, 1,003 validation, and 2,005 test examples.
- PneumoniaMNIST: binary chest X-ray classification, with 4,708 training, 524 validation, and 624 test examples.

Official predefined train, validation, and test splits were used without re-splitting.

Corrupted evaluation sets were obtained from MedMNIST-C. BloodMNIST-C contained 11 corruption types, DermaMNIST-C contained 15, and PneumoniaMNIST-C contained 13. Each corruption was evaluated at five severity levels.

BloodMNIST-C included the task-specific corruptions `stain_deposit` and `bubble`. DermaMNIST-C included `black_corner` and `characters`. PneumoniaMNIST-C contained no task-specific corruption family and was included as a contrast dataset.

### Model

All experiments used ResNet-18.

For the pretrained condition, the network was initialized with the torchvision `IMAGENET1K_V1` ResNet-18 weights. For the scratch condition, the same ResNet-18 architecture was initialized randomly using `weights=None`.

The final fully connected layer was replaced to match the number of classes in each dataset.

No alternative architecture was used in the primary experiment.

### Input preprocessing

All images were loaded at 224 × 224 resolution and converted to three-channel RGB input using the MedMNIST `as_rgb=True` option.

Both pretrained and scratch models used the same ImageNet normalization:

- mean = [0.485, 0.456, 0.406]
- standard deviation = [0.229, 0.224, 0.225]

Using identical normalization in both conditions avoided introducing a preprocessing difference between the pretrained and scratch models.

No corruption-based data augmentation was used during training.

### Training procedure

Models were trained using AdamW with a batch size of 64.

The pretrained condition used a learning rate of 1 × 10^-4.

The scratch learning rate was selected before the main experiment using a small validation-based search on BloodMNIST seed 0 over:

- 1 × 10^-4
- 3 × 10^-4
- 1 × 10^-3

The selected scratch learning rate was 3 × 10^-4 and was then fixed for all subsequent datasets and seeds.

A cosine annealing learning-rate schedule was used with a maximum training budget of 50 epochs.

Early stopping was based on validation balanced accuracy with patience of 10 epochs. The checkpoint corresponding to the highest validation balanced accuracy was retained for evaluation.

### Random seeds and core experiment

The core experiment used five random seeds:

0, 1, 2, 3, and 4.

For each of the three datasets, both initialization conditions were trained for all five seeds:

3 datasets × 2 initialization conditions × 5 seeds = 30 core training runs.

No completed seed was removed as an outlier.

### Training-set size ablation

To test the effect of data availability, an additional BloodMNIST ablation was conducted using 50%, 25%, and 10% of the original training set.

For each fraction, pretrained and scratch models were trained using seeds 0, 1, and 2.

The 100% setting used the corresponding core experiment models for seeds 0–2.

The reduced training subsets were sampled deterministically using the experiment seed. Validation and test sets were unchanged.

The ablation therefore contributed:

3 reduced fractions × 2 initialization conditions × 3 seeds = 18 additional runs.

Across the core and ablation experiments, 48 training runs were completed.

### Clean-set evaluation

Each selected checkpoint was evaluated on the official clean test split.

The primary clean-set metric was balanced accuracy, which gives equal weight to each class and is therefore less sensitive than ordinary accuracy to class imbalance.

Results were summarized as mean ± standard deviation across seeds.

### Corruption evaluation

For each dataset, every trained core checkpoint was evaluated on all available MedMNIST-C corruptions at severity levels 1 through 5.

Balanced accuracy was retained separately for every corruption and severity level.

Corruptions were also grouped into families including digital, noise, blur, color/intensity, and task-specific corruptions where applicable.

The effect of pretraining was expressed so that positive values indicated an advantage for ImageNet initialization.

For balanced accuracy:

ΔbACC = bACC_pretrained − bACC_scratch.

For balanced error:

ΔBE = BE_scratch − BE_pretrained.

Thus, positive ΔBE indicates lower normalized balanced error for the pretrained condition.

### Balanced error and relative balanced error

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

### rBE denominator handling

A known issue occurs when the denominator of rBE is very small or negative.

For PneumoniaMNIST, `contrast_up` and `gamma_corr_up` produced negative AlexNet rBE denominators. These corruptions were therefore excluded from rBE aggregates according to the preregistered rule.

Corruptions with denominator magnitude below 0.05 were flagged as potentially unstable.

No such exclusion was applied to BE, which remained the primary normalized metric.

Absolute balanced-accuracy degradation from clean to corrupted performance was also reported alongside normalized metrics.

### Primary analysis

The preregistered primary endpoint tested whether the effect of ImageNet initialization differed between task-specific corruptions and pooled noise, blur, and digital corruptions on BloodMNIST and DermaMNIST.

For each corruption, performance was first averaged across severity levels and seeds.

The primary interaction was defined as:

mean ΔBE_task-specific − mean ΔBE_noise/blur/digital.

A positive interaction indicates a larger pretraining advantage for task-specific corruptions.

### Statistical analysis

All main performance results were summarized as mean ± standard deviation over seeds.

For corruption-level robustness comparisons, the corruption type was the unit of pairing.

Two-sided Wilcoxon signed-rank tests were applied to per-corruption ΔBE values within each dataset.

Bootstrap 95% confidence intervals were estimated using 10,000 resamples for family-level ΔBE and for the primary task-specific interaction.

Holm–Bonferroni correction was applied across family-level tests within each dataset.

The significance threshold was α = 0.05.

For the training-fraction and severity hypotheses, direction, effect size, and confidence intervals were emphasized rather than regression p-values.

### Hypothesis-specific analyses

H1 tested whether the ImageNet pretraining benefit was smaller for task-specific corruptions than for pooled noise, blur, and digital corruptions.

H2 tested whether the balanced-accuracy advantage of pretraining increased as the available BloodMNIST training data decreased from 100% to 50%, 25%, and 10%.

H3 tested whether gains in corrupted performance were largely explained by clean-set gains. We compared paired clean and corrupted balanced-accuracy differences and examined the relationship between clean-accuracy gain and robustness advantage.

H4 tested whether the pretrained-minus-scratch balanced-accuracy gap decreased monotonically as corruption severity increased from level 1 to level 5 across dataset × corruption-family cells.

### Software and hardware

Experiments were implemented in Python using PyTorch, torchvision, MedMNIST, MedMNIST-C, scikit-learn, pandas, NumPy, SciPy, and Matplotlib.

Training and evaluation were executed using the PyTorch MPS backend on Apple Silicon when available.

All experiment scripts, analysis code, configuration decisions, and result tables are maintained in the accompanying public repository for reproducibility.

## Results

### Clean and corrupted performance

ImageNet initialization improved clean balanced accuracy on all three datasets, but the effect on corruption performance varied substantially across datasets. On BloodMNIST, the pretrained model achieved 99.02% clean balanced accuracy compared with 98.60% for training from scratch. Under corruption, the gap widened substantially: pretrained models achieved 89.01% mean corrupted balanced accuracy, compared with 72.64% for scratch models.

On DermaMNIST, ImageNet initialization produced a much larger clean-set advantage, with mean clean balanced accuracy of 72.66% for pretrained models and 55.38% for scratch models. However, this advantage did not transfer to corrupted performance. Mean corrupted balanced accuracy was 40.37% for pretrained models and 40.86% for scratch models.

A similar divergence appeared on PneumoniaMNIST. Pretrained models achieved 88.90% clean balanced accuracy compared with 80.87% for scratch models, but under corruption the pretrained models performed worse on average, with 72.19% balanced accuracy compared with 76.54% for scratch models.

These results indicate that improved clean performance from ImageNet initialization did not consistently imply improved corruption robustness (Table 1).

### Corruption-family dependence

The effect of ImageNet initialization depended strongly on corruption family. Family-specific effects are summarized in Table 3 and Figure 1.

On BloodMNIST, pretraining substantially improved robustness to color corruptions, with a mean balanced-accuracy advantage of +30.74 percentage points and a mean ΔBE of +5.53. Task-specific corruptions also favored pretrained models, with a mean balanced-accuracy advantage of +14.04 points and ΔBE of +0.77. Blur corruptions showed a smaller positive effect (+6.46 points), whereas digital corruptions favored scratch training (-7.27 points).

On DermaMNIST, the pattern was similarly heterogeneous. Task-specific corruptions showed the largest pretraining benefit (+21.17 percentage points), followed by color corruptions (+16.29 points). In contrast, blur (-18.22 points), digital (-14.85 points), and noise (-7.60 points) favored scratch-trained models.

PneumoniaMNIST contained no task-specific corruptions. Color corruptions favored pretraining (+9.58 percentage points), whereas noise (-21.01 points), digital (-10.69 points), and blur (-8.57 points) favored scratch training.

The preregistered H1 predicted that the benefit of ImageNet initialization would be smaller for task-specific corruptions than for pooled noise, blur, and digital corruptions. The observed direction was reversed. The task-specific versus pooled interaction was +1.0009 ΔBE on BloodMNIST and +0.5657 on DermaMNIST. The bootstrap 95% confidence interval for the interaction included zero on BloodMNIST [-0.0094, 2.0111] but excluded zero on DermaMNIST [0.4799, 0.6604]. Therefore, H1 was rejected.

### Training-set size ablation

The benefit of ImageNet initialization increased as the available BloodMNIST training data decreased (Figure 2).

At 100% of the training set, the pretrained-minus-scratch balanced-accuracy gap was +0.47 percentage points (95% CI: -0.19 to +1.29). At 50% of the training set, the gap increased to +0.79 points (95% CI: +0.69 to +0.96). At 25%, it increased further to +1.03 points (95% CI: +0.90 to +1.17). At 10% of the training set, the gap increased sharply to +3.72 points (95% CI: +2.36 to +6.09).

The increase was monotonic as the training fraction decreased, and the confidence intervals for the 10% and 100% settings did not overlap. Therefore, H2 was supported.

### Clean accuracy versus corruption robustness

The relationship between clean accuracy gain and corruption robustness gain did not follow the preregistered expectation (Figure 3).

On BloodMNIST, the mean clean balanced-accuracy gain from pretraining was only +0.42 percentage points, whereas the mean corrupted-set gain was +16.38 points. On DermaMNIST, the clean gain was +17.28 points, but the corrupted-set gain was -0.48 points. On PneumoniaMNIST, the clean gain was +8.03 points, while the corrupted-set gain was -4.35 points.

Across the 15 dataset-seed pairs, clean-accuracy gain was negatively associated with corrupted-accuracy gain (Pearson r = -0.63; Spearman ρ = -0.46). Clean gain was also negatively associated with BE advantage (Pearson r = -0.79; Spearman ρ = -0.63).

Thus, the corrupted-set advantage of ImageNet initialization was not explained by the corresponding clean-set advantage. H3 was rejected.

### Severity dependence

The preregistered H4 predicted that the pretrained-minus-scratch balanced-accuracy gap would decrease monotonically as corruption severity increased (Figure 4).

This pattern was uncommon. Only 2 of 13 dataset × corruption-family cells (15.4%) showed a strictly monotonic decrease across severity levels 1 through 5. These were DermaMNIST blur and PneumoniaMNIST digital corruptions.

At the dataset level, BloodMNIST showed the opposite tendency: the overall pretraining advantage increased from +4.44 percentage points at severity 1 to +19.93 points at severity 5, with the largest gap occurring at severity 3. DermaMNIST shifted from a small pretrained advantage at severity 1 (+1.79 points) to a scratch advantage at severity 5 (-2.84 points). PneumoniaMNIST also increasingly favored scratch training as severity increased, from -0.33 points at severity 1 to -7.76 points at severity 5.

Because monotonic decreases occurred in only a small minority of dataset × family cells, H4 was rejected.
Paired-seed bootstrap confidence intervals showed that many family-specific effects were clearly separated from zero at several severity levels, but the direction and magnitude of the effect varied substantially across families. The addition of uncertainty estimates did not change the preregistered H4 conclusion.

### Summary of hypothesis tests

Of the four preregistered hypotheses, only H2 was supported. H1, H3, and H4 were rejected (Table 2). Overall, the results show that ImageNet initialization does not produce a uniform corruption-robustness benefit in medical image classification. Its effect depends strongly on dataset, corruption family, training-set size, and corruption severity.

## Discussion

This study examined how much corruption robustness in medical image classification can be attributed to ImageNet initialization rather than to learning from the target medical dataset alone. The main finding is that ImageNet pretraining did not provide a uniform robustness advantage. Its effect depended strongly on the dataset, corruption family, training-set size, and corruption severity.

Three of the four preregistered hypotheses were rejected. These negative results are informative because they challenge several simple interpretations of transfer learning. In particular, higher clean accuracy did not reliably imply higher corruption robustness, task-specific corruptions did not show the smaller pretraining benefit that had been predicted, and increasing corruption severity did not produce a consistent reduction in the pretraining advantage.

### Pretraining robustness is corruption-family dependent

The strongest recurring pattern was heterogeneity across corruption families.

On BloodMNIST, ImageNet initialization provided large improvements for color/intensity and task-specific corruptions, while digital corruptions favored training from scratch. On DermaMNIST, task-specific and color corruptions again favored pretrained models, whereas noise, blur, and digital corruptions favored scratch models. PneumoniaMNIST showed a related division: color/intensity corruptions favored pretraining, while noise, blur, and digital corruptions generally favored scratch training.

This suggests that robustness should not be treated as a single model property. A model can be more robust to one family of distribution shifts while simultaneously being less robust to another.

The result is especially notable for task-specific corruptions. H1 predicted that the ImageNet pretraining benefit would be smaller for medical task-specific corruptions than for generic noise, blur, and digital corruptions. Instead, the observed direction was reversed on both BloodMNIST and DermaMNIST. The interaction was particularly clear on DermaMNIST.

One possible interpretation is that ImageNet initialization provides representations that remain useful under certain structured visual changes, even when those changes are specific to medical imagery. However, the present experiments do not identify the representation-level mechanism responsible for this behavior. Feature-space analyses would be required to test such an explanation directly.

### Clean accuracy is not a reliable proxy for corruption robustness

The results provide particularly strong evidence against interpreting clean accuracy as a proxy for robustness.

BloodMNIST showed only a small clean balanced-accuracy advantage from ImageNet initialization, yet pretraining produced a large mean advantage under corruption. DermaMNIST showed the opposite pattern: pretraining produced a very large improvement on clean images but almost no average corrupted-set advantage. PneumoniaMNIST similarly showed a substantial clean improvement while scratch models performed better on the corrupted test sets on average.

The relationship across dataset-seed pairs was therefore negative rather than positive.

This distinction matters when transfer learning is evaluated only using standard clean test sets. A pretrained model may appear substantially better under conventional evaluation while providing little or no advantage after clinically plausible or acquisition-related image degradation. Conversely, a relatively small clean improvement can coexist with a large robustness improvement, as observed on BloodMNIST.

These findings indicate that clean-set performance and corruption robustness should be evaluated separately rather than assuming that one predicts the other.

### ImageNet pretraining becomes more valuable in the low-data regime

H2 was the only preregistered hypothesis that was supported.

On BloodMNIST, the clean balanced-accuracy advantage of ImageNet initialization increased monotonically as the available training data decreased. The gap was approximately 0.47 percentage points with the full training set but increased to approximately 3.72 points when only 10% of the training data was available.

This result is consistent with the conventional motivation for transfer learning: pretrained representations are most useful when the target dataset contains limited supervision.

Importantly, the result also provides context for the core robustness findings. The usefulness of pretraining cannot be summarized independently of data availability. A model initialization that has only a small effect when thousands of labeled examples are available may become substantially more important when the training set is restricted.

The present ablation was conducted only on BloodMNIST, so the same relationship should not automatically be assumed for DermaMNIST or PneumoniaMNIST.

### Severity does not produce a universal pretraining trend

H4 predicted that the advantage of ImageNet initialization would decrease monotonically as corruption severity increased. This pattern appeared in only 2 of the 13 dataset-by-family cells.

Instead, several qualitatively different severity responses were observed.

BloodMNIST often showed an increasing pretraining advantage as severity increased before reaching a plateau or declining slightly at the highest severity. DermaMNIST showed families in which an initial pretrained advantage diminished or reversed, but also families with stable or non-monotonic behavior. PneumoniaMNIST increasingly favored scratch training for several corruption types as severity increased.

Therefore, corruption severity cannot be interpreted simply as scaling the same robustness difference between pretrained and scratch models. Increasing severity can change both the magnitude and, in some cases, the direction of the initialization effect.

This observation also argues for retaining per-severity results rather than reporting only a single corruption average.

### Why might ImageNet initialization help some corruptions but hurt others?

The present experiments were designed to establish the existence and structure of the effect rather than to identify its internal mechanism. Nevertheless, several explanations motivate future investigation.

ImageNet pretraining may encourage sensitivity to particular textures, color statistics, edges, or spatial frequency patterns. Some corruption families may preserve useful pretrained features, whereas others may disrupt them disproportionately. Scratch-trained models may develop different dataset-specific cues that happen to remain stable under certain noise, blur, or digital transformations.

Differences in medical image modality may also matter. BloodMNIST and DermaMNIST contain color information, whereas PneumoniaMNIST consists of chest radiographs. However, only three datasets were evaluated, so the current study cannot establish a modality-level explanation.

These possibilities should therefore be viewed as hypotheses for future representation-level analysis rather than conclusions from the current results.

### Implications for robustness evaluation

The findings have several practical implications for evaluating medical image classifiers.

First, reporting only clean test accuracy can conceal substantial differences in behavior under corruption. Second, averaging all corruption types into one number can hide strong family-specific reversals. Third, the effectiveness of transfer learning depends on training-set size and therefore should be evaluated under realistic data regimes. Finally, severity-resolved evaluation can reveal effects that disappear when severity levels are averaged.

For robustness benchmarking, these results support reporting clean performance, corruption-family performance, per-severity behavior, and normalized robustness metrics together.

### Limitations

Several limitations constrain the conclusions of this study.

First, all experiments used ResNet-18. The observed effects may differ for larger convolutional networks, vision transformers, or modern self-supervised models.

Second, ImageNet supervised initialization was the only form of pretraining evaluated. The conclusions therefore do not directly extend to self-supervised, foundation-model, or medical-domain pretraining.

Third, the study included only three MedMNIST datasets. This was sufficient to expose substantial heterogeneity but not to establish general conclusions across medical imaging modalities.

Fourth, the training-fraction ablation was performed only on BloodMNIST and used three seeds at the reduced fractions, compared with five seeds in the core experiment.

Fifth, corruption benchmarks provide controlled distribution shifts but cannot reproduce the full complexity of real clinical deployment shifts.

Sixth, some MedMNIST-C rBE denominators are small or negative. We handled this using the preregistered exclusion and flagging rules and treated BE as the primary normalized metric, but this illustrates a limitation of relative normalized metrics.

Finally, DermaMNIST has previously reported label-quality concerns, which should be considered when interpreting results from that dataset.

## Conclusion

ImageNet initialization substantially improves clean medical image classification in several settings, but its contribution to corruption robustness is not uniform.

Across BloodMNIST, DermaMNIST, and PneumoniaMNIST, the effect of pretraining depended strongly on corruption family and severity. Task-specific corruptions often benefited from pretraining more than generic corruptions, contrary to the preregistered hypothesis. Clean-accuracy improvements did not predict robustness improvements, and in some cases large clean gains coexisted with worse corrupted performance. In contrast, the advantage of pretraining increased clearly as the amount of labeled BloodMNIST training data decreased.

Together, these findings suggest that the question is not simply whether ImageNet pretraining improves robustness, but which distribution shifts inherit useful robustness from pretraining, under which data regimes, and at what corruption severity.

## Figure Captions

**Figure 1. Corruption-family dependence of ImageNet pretraining.** Mean pretrained-minus-scratch balanced-accuracy difference for each corruption family on BloodMNIST, DermaMNIST, and PneumoniaMNIST. Positive values indicate an advantage for ImageNet initialization; negative values indicate an advantage for training from scratch.

**Figure 2. Effect of training-set size on the benefit of ImageNet initialization.** Pretrained-minus-scratch clean balanced-accuracy gap on BloodMNIST using 100%, 50%, 25%, and 10% of the training data. Error bars show bootstrap 95% confidence intervals over paired seed-level differences.

**Figure 3. Clean accuracy gain versus corrupted accuracy gain.** Each point represents a paired pretrained-versus-scratch comparison for one dataset and random seed. Positive values indicate an advantage for ImageNet initialization. Clean-set improvements were not positively associated with corrupted-set improvements.

**Figure 4. Pretraining advantage across corruption severity.** Pretrained-minus-scratch balanced-accuracy difference across severity levels 1–5, stratified by dataset and corruption family. Error bars show paired-seed bootstrap 95% confidence intervals based on five matched pretrained-versus-scratch seeds. Positive values favor ImageNet initialization; negative values favor training from scratch.

## Table Captions

**Table 1. Clean and corrupted performance by dataset and initialization condition.** Values are mean ± standard deviation across five random seeds. Balanced error (BE) is the primary normalized corruption metric; relative balanced error (rBE) is secondary.

**Table 2. Summary of preregistered hypotheses and outcomes.** H1, H3, and H4 were rejected according to the preregistered decision rules; H2 was supported.

**Table 3. Corruption-family effects of ImageNet initialization.** ΔbACC is pretrained minus scratch balanced accuracy in percentage points. ΔBE is defined as BE_scratch − BE_pretrained. Positive values therefore favor ImageNet initialization.

## References

1. Hendrycks D, Dietterich T. Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. International Conference on Learning Representations (ICLR), 2019.

2. Hendrycks D, Lee K, Mazeika M. Using Pre-Training Can Improve Model Robustness and Uncertainty. Proceedings of the 36th International Conference on Machine Learning (ICML), PMLR 97:2712–2721, 2019.

3. Raghu M, Zhang C, Kleinberg J, Bengio S. Transfusion: Understanding Transfer Learning for Medical Imaging. Advances in Neural Information Processing Systems 32, 2019.

4. Yang J, Shi R, Wei D, et al. MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification. Scientific Data. 2023;10:41. doi:10.1038/s41597-022-01721-8.

5. Di Salvo F, Doerrich S, Ledig C. MedMNIST-C: Comprehensive Benchmark and Improved Classifier Robustness by Simulating Realistic Image Corruptions. ADSMI Workshop at MICCAI 2024. arXiv:2406.17536.
