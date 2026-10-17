# Introduction

Deep neural networks for medical image classification are commonly initialized from models pretrained on large natural-image datasets such as ImageNet. Transfer learning is attractive because medical datasets are often smaller than general-purpose computer-vision datasets, and pretrained representations can improve optimization and data efficiency. However, the value of ImageNet pretraining in medical imaging is not uniform. Prior work has shown that transfer from natural images can provide only limited or highly context-dependent benefits on some medical imaging tasks, raising questions about which aspects of model performance are actually inherited from pretraining [3,6,7].

A separate concern is robustness to distribution shift. Standard test accuracy measures performance on images drawn from approximately the same distribution as the training data, but deployed medical images may differ because of acquisition artifacts, noise, compression, changes in contrast, blur, staining variation, or other image degradations. Robustness benchmarks based on controlled corruptions provide a reproducible way to study this problem.

The ImageNet-C benchmark established a widely used framework for evaluating model robustness to common image corruptions across multiple severity levels [1]. Subsequent work showed that pretraining can influence robustness and uncertainty in natural-image settings, suggesting that pretrained representations may contribute properties beyond conventional clean-set accuracy [2].

Whether the same relationship holds in medical imaging is less clear. Medical images differ substantially from ImageNet images in texture, color statistics, semantic structure, acquisition process, and task definition. A pretrained model may therefore inherit useful invariances from ImageNet, but it may also inherit sensitivities that are poorly matched to medical image corruptions.

MedMNIST provides a standardized collection of biomedical image-classification datasets spanning multiple modalities and tasks [4]. MedMNIST-C extends this benchmark with generic, task-specific, and modality-motivated corruptions at multiple severity levels, enabling controlled robustness evaluation across medical imaging datasets [5].

Existing MedMNIST-C evaluations primarily establish benchmark performance and examine robustness-enhancing interventions such as domain-informed augmentation [5]. However, corruption performance can reflect several sources simultaneously, including architecture, training data, augmentation, optimization, and initialization. This makes it difficult to isolate the contribution of ImageNet initialization itself.

In this work, we study that narrower question through a controlled comparison of ImageNet-pretrained and randomly initialized ResNet-18 models. Architecture, data splits, preprocessing, optimizer family, scheduling, stopping criteria, model selection, and evaluation procedures are matched across conditions, while initialization and the condition-specific learning rate differ. The learning-rate asymmetry is addressed transparently in the Methods and examined with a post-hoc symmetric sensitivity check.

We evaluate BloodMNIST, DermaMNIST, and PneumoniaMNIST using their corresponding MedMNIST-C corruption benchmarks. All three datasets include corruption families that allow the effect of initialization to be examined across generic and medically motivated distribution shifts. BloodMNIST and DermaMNIST were the datasets included in the preregistered task-specific-versus-generic H1 contrast, while PneumoniaMNIST task-specific gamma corruptions are analyzed descriptively.

Our analysis addresses four questions. First, does the contribution of ImageNet initialization depend on corruption family? Second, does the benefit of pretraining increase when medical training data are limited? Third, are improvements under corruption largely explained by improvements on clean data? Fourth, does the effect of pretraining change systematically with corruption severity?

The main finding is that ImageNet initialization does not confer a uniform corruption-robustness advantage. Its effect depends strongly on corruption family, dataset, training-set size, and severity. Task-specific corruptions often benefited from pretraining more than the pooled generic corruptions used in the preregistered H1 contrast, the relationship between clean gains and robustness gains was strongly dataset-dependent, and the value of pretraining increased clearly in the low-data BloodMNIST regime.

These results suggest that medical image robustness should not be inferred from clean performance or from the presence of ImageNet pretraining alone. Instead, the contribution of pretraining should be evaluated under the specific distribution shifts and data regimes relevant to the target imaging setting.


# Related Work

## Corruption robustness benchmarks

Controlled corruption benchmarks provide a standardized framework for measuring performance under non-adversarial image degradation. Hendrycks and Dietterich introduced ImageNet-C and related benchmarks to evaluate classifiers under common corruptions and perturbations rather than worst-case adversarial attacks [1]. ImageNet-C applies multiple corruption types at several severity levels and has become a common reference for evaluating distribution-shift robustness.

The same general framework has increasingly been applied to medical imaging, where acquisition artifacts and modality-specific shifts can differ substantially from those encountered in natural images [5,9].

## MedMNIST and MedMNIST-C

MedMNIST v2 provides a standardized collection of lightweight biomedical image-classification datasets spanning multiple medical imaging modalities, dataset sizes, and prediction tasks [4]. The benchmark facilitates reproducible comparisons using common APIs and predefined data splits.

MedMNIST-C extends this collection specifically for corruption robustness. Di Salvo, Doerrich, and Ledig introduced corruptions designed to simulate both generic image degradation and artifacts motivated by particular medical datasets and imaging modalities [5]. The benchmark covers 12 two-dimensional MedMNIST datasets and nine imaging modalities and evaluates corruptions at five severity levels [5].

Their work demonstrated substantial vulnerability of medical classifiers to corruption and showed that targeted, domain-informed augmentation can improve robustness [5]. Our study uses the MedMNIST-C evaluation framework but asks a different question: rather than proposing a robustness intervention, we examine how model initialization itself changes corruption performance.

## ImageNet pretraining and robustness

ImageNet pretraining is widely used for transfer learning. Beyond data efficiency and optimization, pretraining has also been associated with robustness-related benefits. Hendrycks, Lee, and Mazeika showed that pretraining can improve robustness and uncertainty across several settings, including label corruption, adversarial examples, out-of-distribution detection, and calibration [2].

These findings motivate the possibility that robustness may be partially inherited from the source pretraining task. However, such conclusions cannot automatically be transferred to medical imaging because of the domain gap between natural and medical images.

Our study therefore does not test the broad proposition that pretraining can ever improve robustness. Instead, it asks a narrower question: how does supervised ImageNet initialization affect MedMNIST-C corruption performance relative to training the same architecture from scratch, and how does that effect vary across corruption families, data availability, and severity?

## Transfer learning in medical imaging

The usefulness of natural-image pretraining for medical imaging has been debated. Raghu et al. found that ImageNet transfer could provide surprisingly limited performance benefits in some medical imaging settings [3]. Later work showed that transfer gains depend on factors including target-data size, source-target distance, architecture, and feature reuse [6], while broader benchmarking has also demonstrated that alternative pretraining strategies can behave differently across medical tasks [7].

Together, this literature highlights an important distinction between conventional predictive performance and other properties that may be affected by initialization [3,6,7]. Even when clean-set gains are small, pretraining could still alter robustness; conversely, large clean-set improvements might not guarantee better performance under image degradation.

Medical-domain pretraining approaches such as Models Genesis further show that source-domain choice and self-supervised objectives can materially affect transfer behavior, especially when natural-image pretraining is poorly matched to the target modality [8]. Our experiments examine a narrower comparison by measuring clean performance and corruption robustness separately for supervised ImageNet initialization versus random initialization.

## Position of the present study

The present study sits at the intersection of medical transfer learning and corruption robustness.

Rather than comparing different robustness algorithms, architectures, or augmentation strategies, we perform a matched pretrained-versus-scratch comparison using the same ResNet-18 architecture and evaluation framework. This design allows us to examine the contribution associated with ImageNet initialization while keeping most of the training and evaluation pipeline fixed.

Our focus is therefore not the broad claim that pretraining improves robustness, but the narrower question of when ImageNet initialization helps, when it does not, and how its effect changes across medical corruption families, training-data regimes, and corruption severity.
