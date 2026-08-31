# Discussion

This study examined how much corruption robustness in medical image classification can be attributed to ImageNet initialization rather than to learning from the target medical dataset alone. The main finding is that ImageNet pretraining did not provide a uniform robustness advantage. Its effect depended strongly on the dataset, corruption family, training-set size, and corruption severity.

Three of the four preregistered hypotheses were rejected. These negative results are informative because they challenge several simple interpretations of transfer learning. In particular, higher clean accuracy did not reliably imply higher corruption robustness, task-specific corruptions did not show the smaller pretraining benefit that had been predicted, and increasing corruption severity did not produce a consistent reduction in the pretraining advantage.

## Pretraining robustness is corruption-family dependent

The strongest recurring pattern was heterogeneity across corruption families.

On BloodMNIST, ImageNet initialization provided large improvements for color/intensity and task-specific corruptions, while digital corruptions favored training from scratch. On DermaMNIST, task-specific and color corruptions again favored pretrained models, whereas noise, blur, and digital corruptions favored scratch models. PneumoniaMNIST showed a related division: color/intensity corruptions favored pretraining, while noise, blur, and digital corruptions generally favored scratch training.

This suggests that robustness should not be treated as a single model property. A model can be more robust to one family of distribution shifts while simultaneously being less robust to another.

The result is especially notable for task-specific corruptions. H1 predicted that the ImageNet pretraining benefit would be smaller for medical task-specific corruptions than for generic noise, blur, and digital corruptions. Instead, the observed direction was reversed on both BloodMNIST and DermaMNIST. The interaction was particularly clear on DermaMNIST.

One possible interpretation is that ImageNet initialization provides representations that remain useful under certain structured visual changes, even when those changes are specific to medical imagery. However, the present experiments do not identify the representation-level mechanism responsible for this behavior. Feature-space analyses would be required to test such an explanation directly.

## Clean accuracy is not a reliable proxy for corruption robustness

The results provide particularly strong evidence against interpreting clean accuracy as a proxy for robustness.

BloodMNIST showed only a small clean balanced-accuracy advantage from ImageNet initialization, yet pretraining produced a large mean advantage under corruption. DermaMNIST showed the opposite pattern: pretraining produced a very large improvement on clean images but almost no average corrupted-set advantage. PneumoniaMNIST similarly showed a substantial clean improvement while scratch models performed better on the corrupted test sets on average.

The relationship across dataset-seed pairs was therefore negative rather than positive.

This distinction matters when transfer learning is evaluated only using standard clean test sets. A pretrained model may appear substantially better under conventional evaluation while providing little or no advantage after clinically plausible or acquisition-related image degradation. Conversely, a relatively small clean improvement can coexist with a large robustness improvement, as observed on BloodMNIST.

These findings indicate that clean-set performance and corruption robustness should be evaluated separately rather than assuming that one predicts the other.

## ImageNet pretraining becomes more valuable in the low-data regime

H2 was the only preregistered hypothesis that was supported.

On BloodMNIST, the clean balanced-accuracy advantage of ImageNet initialization increased monotonically as the available training data decreased. The gap was approximately 0.47 percentage points with the full training set but increased to approximately 3.72 points when only 10% of the training data was available.

This result is consistent with the conventional motivation for transfer learning: pretrained representations are most useful when the target dataset contains limited supervision.

Importantly, the result also provides context for the core robustness findings. The usefulness of pretraining cannot be summarized independently of data availability. A model initialization that has only a small effect when thousands of labeled examples are available may become substantially more important when the training set is restricted.

The present ablation was conducted only on BloodMNIST, so the same relationship should not automatically be assumed for DermaMNIST or PneumoniaMNIST.

## Severity does not produce a universal pretraining trend

H4 predicted that the advantage of ImageNet initialization would decrease monotonically as corruption severity increased. This pattern appeared in only 2 of the 13 dataset-by-family cells.

Instead, several qualitatively different severity responses were observed.

BloodMNIST often showed an increasing pretraining advantage as severity increased before reaching a plateau or declining slightly at the highest severity. DermaMNIST showed families in which an initial pretrained advantage diminished or reversed, but also families with stable or non-monotonic behavior. PneumoniaMNIST increasingly favored scratch training for several corruption types as severity increased.

Therefore, corruption severity cannot be interpreted simply as scaling the same robustness difference between pretrained and scratch models. Increasing severity can change both the magnitude and, in some cases, the direction of the initialization effect.

This observation also argues for retaining per-severity results rather than reporting only a single corruption average.

## Why might ImageNet initialization help some corruptions but hurt others?

The present experiments were designed to establish the existence and structure of the effect rather than to identify its internal mechanism. Nevertheless, several explanations motivate future investigation.

ImageNet pretraining may encourage sensitivity to particular textures, color statistics, edges, or spatial frequency patterns. Some corruption families may preserve useful pretrained features, whereas others may disrupt them disproportionately. Scratch-trained models may develop different dataset-specific cues that happen to remain stable under certain noise, blur, or digital transformations.

Differences in medical image modality may also matter. BloodMNIST and DermaMNIST contain color information, whereas PneumoniaMNIST consists of chest radiographs. However, only three datasets were evaluated, so the current study cannot establish a modality-level explanation.

These possibilities should therefore be viewed as hypotheses for future representation-level analysis rather than conclusions from the current results.

## Implications for robustness evaluation

The findings have several practical implications for evaluating medical image classifiers.

First, reporting only clean test accuracy can conceal substantial differences in behavior under corruption. Second, averaging all corruption types into one number can hide strong family-specific reversals. Third, the effectiveness of transfer learning depends on training-set size and therefore should be evaluated under realistic data regimes. Finally, severity-resolved evaluation can reveal effects that disappear when severity levels are averaged.

For robustness benchmarking, these results support reporting clean performance, corruption-family performance, per-severity behavior, and normalized robustness metrics together.

## Limitations

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
