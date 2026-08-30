from medmnist import PneumoniaMNIST
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = PneumoniaMNIST(
    split="train",
    transform=transform,
    download=True,
    size=224
)

val_dataset = PneumoniaMNIST(
    split="val",
    transform=transform,
    download=True,
    size=224
)

test_dataset = PneumoniaMNIST(
    split="test",
    transform=transform,
    download=True,
    size=224
)

print("Train:", len(train_dataset))
print("Val:", len(val_dataset))
print("Test:", len(test_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

images, labels = next(iter(train_loader))

print("Batch image shape:", images.shape)
print("Batch label shape:", labels.shape)

pretrained_model = resnet18(
    weights=ResNet18_Weights.IMAGENET1K_V1
)

pretrained_model.fc = nn.Linear(
    pretrained_model.fc.in_features,
    2
)

print("Pretrained model ready")

scratch_model = resnet18(weights=None)

scratch_model.fc = nn.Linear(
    scratch_model.fc.in_features,
    2
)

print("Scratch model ready")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Device:", device)

pretrained_model = pretrained_model.to(device)
scratch_model = scratch_model.to(device)

images = images.to(device)

pretrained_model.eval()
scratch_model.eval()

with torch.no_grad():
    pretrained_output = pretrained_model(images)
    scratch_output = scratch_model(images)

print("Pretrained output shape:", pretrained_output.shape)
print("Scratch output shape:", scratch_output.shape)

print("\nSMOKE TEST PASSED ✅")