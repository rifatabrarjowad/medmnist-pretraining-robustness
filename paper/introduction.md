# Introduction

Deep neural networks for medical image classification are commonly initialized from models pretrained on large natural-image datasets such as ImageNet. Transfer learning is attractive because medical datasets are often smaller than general-purpose computer-vision datasets, and pretrained representations can improve optimization and data efficiency. However, the value of ImageNet pretraining in medical imaging is not uniform. Prior work has shown that transfer from natural images can provide only limited benefits on some medical imaging tasks, raising questions about which aspects of model performance are actually inherited from pretraining.

A separate concern is robustness to distribution shift. Standard test accuracy measures performance on images drawn from approximately the same distribution as the training data, but deployed medical images may differ because of acquisition artifacts, noise, compression, changes in contrast, blur, staining variation, or other image degradations. Robustness benchmarks based on controlled corruptions provide a reproducible way to study this problem.

The ImageNet-C benchmark established a widely used framework for evaluating model robustness to common image corruptions at multiple severity levels. Subsequent work showed that pretraining can improve robustness and uncertainty in natural-image settings, suggesting that pretrained representations may contribute properties beyond conventional clean-set accuracy.

Whether the same relationship holds in medical imaging is less clear. Medical images differ substantially from ImageNet images in texture, color statistics, semantic structure, acquisition process, and task definition. A pretrained model may therefore inherit useful invariances from ImageNet, but it may also inherit sensitivities that are poorly matched to medical image corruptions.

MedMNIST provides a standardized collection of biomedical image-classification datasets spanning multiple modalities and tasks. MedMNIST-C extends this benchmark by introducing generic as well as task- and modality-specific corruptions at multiple severity levels, enabling controlled robustness evaluation across medical imaging datasets.

Existing MedMNIST-C evaluations primarily establish corruption benchmarks and robustness-enhancing augmentation strategies. However, corruption performance can reflect several sources simultaneously: architecture, training data, optimization, augmentation, and initialization. This makes it difficult to determine how much robustness is specifically inherited from ImageNet pretraining.

In this work, we isolate the effect of initialization through a controlled comparison of ImageNet-pretrained and randomly initialized ResNet-18 models. Architecture, preprocessing, optimization, model-selection procedure, and evaluation protocol are held fixed, while initialization is varied.

We evaluate BloodMNIST, DermaMNIST, and PneumoniaMNIST using their corresponding MedMNIST-C corruption benchmarks. These datasets provide complementary conditions: BloodMNIST and DermaMNIST contain task-specific corruptions, whereas PneumoniaMNIST provides a contrast setting without task-specific corruptions.

Our analysis addresses four questions. First, does the contribution of ImageNet initialization depend on corruption family? Second, does the benefit of pretraining increase when medical training data are limited? Third, are improvements under corruption largely explained by improvements on clean data? Fourth, does the effect of pretraining change systematically with corruption severity?

The main finding is that ImageNet initialization does not confer a uniform corruption-robustness advantage. Its effect depends strongly on corruption family, dataset, training-set size, and severity. In particular, task-specific corruptions often benefited from pretraining more than generic corruptions, clean-accuracy gains did not predict corruption-robustness gains, and the value of pretraining increased clearly in the low-data BloodMNIST regime.

These results suggest that medical image robustness should not be inferred from clean performance or from the presence of ImageNet pretraining alone. Instead, the contribution of pretraining should be evaluated under the specific distribution shifts that matter for the target imaging setting.


# Related Work

## Corruption robustness benchmarks

Controlled corruption benchmarks provide a standardized framework for measuring performance under non-adversarial image degradation. Hendrycks and Dietterich introduced ImageNet-C and related benchmarks to evaluate classifiers under common corruptions and perturbations rather than worst-case adversarial attacks. ImageNet-C applies multiple corruption types at several severity levels and has become a common reference for evaluating distribution-shift robustness.

The same general framework has increasingly been applied to medical imaging, where acquisition artifacts and modality-specific shifts can differ substantially from those encountered in natural images.

## MedMNIST and MedMNIST-C

MedMNIST v2 provides a standardized collection of lightweight biomedical image-classification datasets spanning multiple medical imaging modalities, dataset sizes, and prediction tasks. The benchmark facilitates reproducible comparisons using common APIs and predefined data splits.

MedMNIST-C extends this collection specifically for corruption robustness. Di Salvo, Doerrich, and Ledig introduced corruptions designed to simulate both generic image degradation and artifacts motivated by particular medical datasets and imaging modalities. The benchmark covers 12 two-dimensional MedMNIST datasets and nine imaging modalities and evaluates corruptions at five severity levels.

Their work demonstrated substantial vulnerability of medical classifiers to corruption and showed that targeted, domain-informed augmentation can improve robustness. Our study uses the MedMNIST-C evaluation framework but asks a different question: rather than proposing a robustness intervention, we isolate how model initialization itself affects corruption performance.

## ImageNet pretraining and robustness

ImageNet pretraining is widely used for transfer learning. Beyond data efficiency and optimization, pretraining has also been associated with robustness-related benefits. Hendrycks, Lee, and Mazeika showed that pretraining can improve robustness and uncertainty across several settings, including label corruption, adversarial examples, out-of-distribution detection, and calibration.

These findings motivate the possibility that corruption robustness may be partially inherited from the source pretraining task. However, such conclusions cannot automatically be transferred to medical imaging because of the domain gap between natural and medical images.

Our study therefore does not test whether pretraining can ever improve robustness; that phenomenon has already been established in broader computer-vision research. Instead, we examine how much of MedMNIST-C robustness is attributable specifically to ImageNet initialization and whether that contribution changes across medical datasets, corruption families, data availability, and severity.

## Transfer learning in medical imaging

The usefulness of natural-image pretraining for medical imaging has been debated. Raghu et al. studied transfer learning on medical imaging tasks and found that ImageNet transfer sometimes provided surprisingly limited performance benefits. Their analysis emphasized that the gains associated with transfer may not always arise from sophisticated reuse of high-level ImageNet features.

This work highlights an important distinction between conventional predictive performance and other properties inherited through initialization. Even when clean-set gains are small, pretraining could still affect robustness; conversely, large clean-set improvements might not guarantee robust performance under image degradation.

Our experiments directly examine this distinction by measuring clean accuracy and corruption robustness separately.

## Position of the present study

The present study sits at the intersection of medical transfer learning and corruption robustness.

Rather than comparing different robustness algorithms, architectures, or augmentation methods, we hold these factors fixed and perform a controlled pretrained-versus-scratch comparison. This allows us to decompose the contribution of ImageNet initialization itself.

Our focus is therefore not the broad claim that pretraining improves robustness, but a narrower question: under which medical image corruptions does ImageNet initialization help, under which does it hurt, and how does that effect depend on data availability and corruption severity?
