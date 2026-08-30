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
