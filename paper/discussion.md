# Discussion

This study examined how much corruption robustness in medical image classification can be attributed to ImageNet initialization rather than to learning from the target medical dataset alone. The central finding is that ImageNet pretraining did not provide a uniform robustness advantage. Its effect depended strongly on the dataset, corruption family, training-set size, and corruption severity.

Three of the four preregistered hypotheses were rejected according to their original directional predictions and analysis definitions. These results challenge several simple interpretations of transfer learning. Task-specific corruptions did not show the smaller pretraining benefit that had been predicted, gains under corruption were not well summarized by clean-set gains alone, and increasing corruption severity did not produce a consistent reduction in the pretraining advantage. In contrast, the benefit of ImageNet initialization increased clearly in the most data-limited BloodMNIST regime.

## Pretraining robustness is corruption-family dependent

The strongest recurring pattern was heterogeneity across corruption families.

On BloodMNIST, ImageNet initialization provided large improvements for color/intensity and task-specific corruptions, while digital corruptions favored training from scratch. On DermaMNIST, task-specific and color corruptions again favored pretrained models, whereas noise, blur, and digital corruptions favored scratch-trained models. PneumoniaMNIST showed a related split: task-specific gamma and color/intensity corruptions favored pretraining, while noise, blur, and digital corruptions generally favored scratch training.

This pattern indicates that robustness should not be treated as a single model property. A model can be more robust to one family of distribution shifts while simultaneously being less robust to another.

The result is especially notable for task-specific corruptions. H1 predicted that the ImageNet pretraining benefit would be smaller for task-specific medical corruptions than for pooled noise, blur, and digital corruptions. Instead, the observed interaction was in the opposite direction on both BloodMNIST and DermaMNIST, with the effect particularly clear on DermaMNIST. PneumoniaMNIST task-specific gamma results showed a similar descriptive tendency, although PneumoniaMNIST was not part of the preregistered H1 contrast.

One possible interpretation is that ImageNet initialization provides representations that remain useful under certain structured visual changes, even when those changes are specific to medical imagery. However, the present experiments do not identify the representation-level mechanism responsible for this behavior. Feature-space, frequency-response, or attribution analyses would be needed to test such explanations directly.

## Clean accuracy and corruption robustness capture different behavior

The results show that clean performance alone is insufficient to characterize robustness.

BloodMNIST showed only a small clean balanced-accuracy advantage from ImageNet initialization, yet pretraining produced a large mean advantage under corruption. DermaMNIST showed the opposite pattern: pretraining produced a large improvement on clean images but almost no average corrupted-set advantage. PneumoniaMNIST similarly showed a substantial clean improvement while scratch-trained models performed better on corrupted data on average.

In the preregistered pooled H3 analysis, the relationship across the 15 dataset-seed pairs was negative: larger clean gains were associated with smaller corrupted-accuracy gains and smaller BE advantages. However, the 15 observations were not independent samples from a common population; they consisted of repeated seeds from only three datasets.

A post-hoc decomposition showed that the pooled negative association was strongly influenced by between-dataset differences. After centering clean gain and robustness gain within each dataset, the association reversed direction and became positive for both corrupted-accuracy gain and BE advantage. Within-dataset correlations were also heterogeneous, with positive associations on DermaMNIST and PneumoniaMNIST but weaker or mixed patterns on BloodMNIST.

This distinction is important. The preregistered H3 result remains rejected because the original pooled analysis did not support the hypothesis that corrupted-performance gains were largely explained by clean-set gains. However, the pooled negative correlation should not be interpreted as evidence that larger clean improvements intrinsically cause worse robustness. Instead, the analyses indicate that clean-set and corruption performance capture related but distinct aspects of model behavior, and their relationship depends strongly on the dataset.

This conclusion is reinforced by the degradation sensitivity analysis. After accounting for both clean performance and chance-level balanced accuracy, pretrained models retained more of their above-chance performance than scratch models on BloodMNIST, but retained less on DermaMNIST and PneumoniaMNIST. The heterogeneous robustness pattern therefore cannot be explained solely by lower scratch-model clean accuracy or by floor effects.

## ImageNet pretraining becomes more valuable in the low-data regime

H2 was the only preregistered hypothesis that was supported.

On BloodMNIST, the clean balanced-accuracy advantage of ImageNet initialization increased monotonically as the available training data decreased. The gap was approximately 0.47 percentage points with the full training set but increased to approximately 3.72 points when only 10% of the training data was available.

The additional corruption analysis showed the same broad low-data pattern, although the intermediate fractions were not strictly monotonic. The mean corrupted-performance advantage was approximately 16.58 percentage points at 100% of the training data, 18.12 points at 50%, 17.07 points at 25%, and 23.40 points at 10%. Thus, the largest robustness benefit also occurred in the most data-limited regime.

A paired comparison between the 100% and 10% conditions showed that reducing the available training data increased the pretraining advantage by approximately 3.25 percentage points on clean data and 6.83 points under corruption, with all three paired seeds showing increases.

This result is consistent with prior evidence that the value of transfer learning can depend strongly on target-dataset size and available supervision [6]. It also extends that intuition beyond clean accuracy by showing that the robustness advantage of pretraining can become larger when labeled medical data are scarce.

The ablation was conducted only on BloodMNIST, and the reduced-data conditions used three paired seeds. The same relationship should therefore not automatically be assumed for DermaMNIST, PneumoniaMNIST, or other medical imaging datasets.

## Severity does not produce a universal pretraining trend

H4 predicted that the advantage of ImageNet initialization would decrease monotonically as corruption severity increased. This pattern appeared in only 2 of the 14 dataset-by-family cells.

Instead, several qualitatively different severity responses were observed.

BloodMNIST showed an overall increase in pretraining advantage from low to moderate severity, followed by a plateau or slight reduction at the highest severity. DermaMNIST shifted from a small pretrained advantage at lower severity to a scratch advantage at higher severity. PneumoniaMNIST increasingly favored scratch training as severity increased at the dataset level.

At the family level, only DermaMNIST blur and PneumoniaMNIST digital corruptions satisfied the preregistered strictly monotonic decreasing pattern. Other families showed increasing, stable, reversing, or non-monotonic effects.

Therefore, corruption severity cannot be interpreted simply as scaling the same robustness difference between pretrained and scratch models. Increasing severity can change both the magnitude and, in some cases, the direction of the initialization effect.

This observation supports reporting severity-resolved results rather than relying only on a single average across all severity levels.

## Why might ImageNet initialization help some corruptions but hurt others?

The present experiments were designed to establish the existence and structure of the effect rather than to identify its internal mechanism. Nevertheless, several explanations motivate future investigation.

ImageNet pretraining may encourage sensitivity to particular textures, color statistics, edges, or spatial-frequency patterns. Some corruption families may preserve useful pretrained features, whereas others may disrupt them disproportionately. Scratch-trained models may develop different dataset-specific cues that happen to remain stable under certain noise, blur, or digital transformations.

Differences in imaging modality may also matter. BloodMNIST and DermaMNIST contain color information, whereas PneumoniaMNIST consists of chest radiographs. However, only three datasets were evaluated, so the current study cannot establish a modality-level explanation.

These possibilities should therefore be viewed as hypotheses for future representation-level analysis rather than conclusions from the current results.

## Implications for robustness evaluation

The findings have several practical implications for evaluating medical image classifiers.

First, reporting only clean test accuracy can conceal substantial differences in behavior under corruption. Second, averaging all corruption types into one number can hide strong family-specific reversals. Third, transfer-learning benefits depend on training-set size and therefore should be evaluated under realistic data regimes. Fourth, severity-resolved evaluation can reveal changes in both magnitude and direction that disappear when severity levels are averaged. Finally, normalized robustness measures should be interpreted alongside raw corrupted performance because normalization can itself be sensitive to clean baselines and benchmark denominators.

For robustness benchmarking, these results support reporting clean performance, corruption-family performance, per-severity behavior, and at least one normalized robustness measure together.

## Limitations

Several limitations constrain the conclusions of this study.

First, all experiments used ResNet-18. The observed effects may differ for larger convolutional networks, vision transformers, or more recent architectures.

Second, supervised ImageNet initialization was the only form of pretraining evaluated. The conclusions therefore do not directly extend to self-supervised, foundation-model, or medical-domain pretraining.

Third, the study included only three MedMNIST datasets. This was sufficient to expose substantial heterogeneity but not to establish general conclusions across medical imaging modalities.

Fourth, the training-fraction ablation was performed only on BloodMNIST and used three seeds at the reduced fractions, compared with five seeds in the core experiment. The implemented reduced-data subsets were deterministic and matched across initialization conditions, but were not explicitly class-stratified as specified in the preregistration, so some seed-level variation in class composition may remain. Bootstrap intervals from the reduced-data analyses should therefore be interpreted as descriptive seed-level uncertainty rather than strong population-level inference.

Fifth, the H3 correlation analyses were based on repeated seeds from only three datasets. The post-hoc within-dataset and dataset-centered analyses were useful for diagnosing between-dataset structure, but they do not provide an independent confirmatory test and should not be used to reinterpret the preregistered H3 result as supported.

Sixth, condition-specific learning rates were used in the core experiments. The original scratch learning-rate search was not symmetric across initialization conditions and used a preprocessing/training-search setup that did not exactly match the final core pipeline. To assess whether this asymmetry materially favored one condition, we performed a post-hoc symmetric learning-rate sensitivity check using the final BloodMNIST preprocessing and training protocol for both pretrained and scratch models over the same grid of 1×10^-4, 3×10^-4, and 1×10^-3. The validation-optimal rates matched those used in the core experiments: 1×10^-4 for pretrained models and 3×10^-4 for scratch models. This result reduces concern that the reported effects arose from arbitrary learning-rate choices, but it does not convert the original asymmetric tuning procedure into a preregistered symmetric comparison.

Seventh, corruption benchmarks provide controlled distribution shifts but cannot reproduce the full complexity of real clinical deployment shifts.

Eighth, some MedMNIST-C rBE denominators are small or negative. We handled these cases using the preregistered exclusion and flagging rules and treated BE as the primary normalized metric, but this illustrates a limitation of relative normalized metrics.


## Conclusion

ImageNet initialization substantially improves clean medical image classification in several settings, but its contribution to corruption robustness is not uniform.

Across BloodMNIST, DermaMNIST, and PneumoniaMNIST, the effect of pretraining depended strongly on corruption family and severity. Task-specific corruptions often benefited from pretraining more than generic corruptions, contrary to the preregistered hypothesis. Clean-set gains and corruption robustness were not interchangeable measures, and the post-hoc decomposition showed that their pooled relationship was strongly influenced by differences between datasets. In contrast, the benefit of pretraining increased clearly in the most data-limited BloodMNIST regime, including under corruption.

Together, these findings suggest that the relevant question is not simply whether ImageNet pretraining improves robustness, but which distribution shifts benefit from pretraining, under which data regimes, and at what corruption severity.
