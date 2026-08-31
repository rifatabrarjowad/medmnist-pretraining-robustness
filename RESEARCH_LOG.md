# Research Log

## 2026-08-30

### Environment setup

- Python 3.11.16
- Created virtual environment `.venv`
- Installed PyTorch, torchvision, medmnist, medmnistc and analysis dependencies.

### Dataset smoke test

- Successfully downloaded and loaded PneumoniaMNIST at 224x224.
- Verified official splits:
  - Train: 4708
  - Validation: 524
  - Test: 624
- Verified image shape: `[1, 224, 224]`
- Verified sample visualization.
- No experimental results collected yet.

### Next step

- Convert grayscale images to 3 channels.
- Verify pretrained and randomly initialized ResNet-18 forward passes.

### Model pipeline smoke test

- Converted PneumoniaMNIST grayscale images to 3-channel input.
- Verified batch shape: [8, 3, 224, 224].
- Loaded ResNet-18 with ImageNet pretrained weights.
- Loaded identical ResNet-18 with random initialization.
- Verified Apple MPS device is available and used.
- Both models completed a forward pass successfully.
- Output shape for both models: [8, 2].
- No experimental training results collected yet.

### Mini-training smoke test

- Successfully trained pretrained ResNet-18 for 3 epochs on PneumoniaMNIST.
- Best validation balanced accuracy: 0.9875.
- Successfully trained randomly initialized ResNet-18 for 3 epochs.
- Best validation balanced accuracy: 0.9699.
- Both training loops completed on Apple MPS.
- Checkpoints were saved locally.
- These are smoke-test results only and will NOT be reported as research findings.
- No scientific comparison is made from these runs.

### Scratch learning-rate fairness search

The preregistered scratch learning-rate search was completed on BloodMNIST
using seed 0.

Results:

| Learning Rate | Best Val bACC | Best Epoch |
| ------------- | ------------: | ---------: |
| 1e-4          |      0.911012 |          3 |
| 3e-4          |      0.984022 |         20 |
| 1e-3          |      0.982944 |         19 |

Selected scratch learning rate: **3e-4**

The selection followed the preregistered rule of choosing the learning rate
with the highest validation balanced accuracy.

The selected value is now frozen for all subsequent scratch-model experiments.

### PneumoniaMNIST core training completed

Completed all 10 preregistered PneumoniaMNIST core runs:

- ImageNet pretrained ResNet-18: seeds 0–4
- Randomly initialized ResNet-18: seeds 0–4

Validation balanced accuracy summary:

- Pretrained: mean ≈ 0.9927, SD ≈ 0.0031
- Scratch: mean ≈ 0.9814, SD ≈ 0.0025

These validation results are training-selection diagnostics only.
They are not the primary research endpoint.

All best-validation checkpoints were saved locally.

## Core robustness analysis and H2 ablation

Completed all preregistered core training runs:

- 3 datasets × 2 initialization conditions × 5 seeds = 30 runs.
- BloodMNIST, DermaMNIST, and PneumoniaMNIST were evaluated on clean test sets and MedMNIST-C corruptions.

Core robustness analysis:

- BloodMNIST showed a strong overall corruption advantage for ImageNet-pretrained models, but the effect varied substantially by corruption family.
- DermaMNIST and PneumoniaMNIST showed mixed effects: pretraining improved several color/intensity and task-specific corruptions, while scratch models performed better on several noise, blur, and digital corruptions.
- The preregistered H1 direction was reversed. Task-specific corruptions showed larger, not smaller, pretraining benefits relative to pooled noise/blur/digital corruptions.
- Bootstrap primary interaction:
  - BloodMNIST: Δ = +1.0009, 95% CI [-0.0094, +2.0111]
  - DermaMNIST: Δ = +0.5657, 95% CI [+0.4799, +0.6604]
- Therefore H1 is rejected under the preregistered decision rule.

Completed H2 training-fraction ablation:

- BloodMNIST fractions: 100%, 50%, 25%, 10%.
- Pretrained vs scratch, seeds 0–2.
- Total ablation runs: 18.
- Total preregistered training runs completed: 48.

H2 clean-test pretraining gap:

- 100%: +0.0047, 95% CI [-0.0019, +0.0129]
- 50%: +0.0079, 95% CI [+0.0069, +0.0096]
- 25%: +0.0103, 95% CI [+0.0090, +0.0117]
- 10%: +0.0372, 95% CI [+0.0236, +0.0609]

The pretraining advantage increased monotonically as training data decreased, and the 10% and 100% confidence intervals did not overlap. H2 is therefore supported under the preregistered decision rule.

No runs or seeds were excluded.
