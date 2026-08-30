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
