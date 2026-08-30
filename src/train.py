import os
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from medmnist import PneumoniaMNIST
from sklearn.metrics import balanced_accuracy_score


SEED = 0
EPOCHS = 3
BATCH_SIZE = 32
LR = 1e-4


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_dataloaders():
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader


def build_model(pretrained=True):
    if pretrained:
        model = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1
        )
    else:
        model = resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    return model


def evaluate(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            labels = labels.squeeze(1).long().to(device)

            outputs = model(images)

            preds = torch.argmax(outputs, dim=1)

            y_true.extend(
                labels.cpu().numpy()
            )

            y_pred.extend(
                preds.cpu().numpy()
            )

    return balanced_accuracy_score(
        y_true,
        y_pred
    )


def train_model(pretrained=True):
    set_seed(SEED)

    device = get_device()

    print("Device:", device)

    train_loader, val_loader = get_dataloaders()

    model = build_model(
        pretrained=pretrained
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    condition = (
        "pretrained"
        if pretrained
        else "scratch"
    )

    print(f"\nTraining condition: {condition}")

    best_val_bacc = -1

    os.makedirs(
        "results/raw",
        exist_ok=True
    )

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)

            labels = labels.squeeze(1).long().to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        avg_loss = (
            running_loss
            / len(train_loader)
        )

        val_bacc = evaluate(
            model,
            val_loader,
            device
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| Loss: {avg_loss:.4f} "
            f"| Val bACC: {val_bacc:.4f}"
        )

        if val_bacc > best_val_bacc:
            best_val_bacc = val_bacc

            checkpoint_path = (
                f"results/raw/"
                f"pneumonia_{condition}_smoke.pt"
            )

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

    print(
        f"Best validation bACC: "
        f"{best_val_bacc:.4f}"
    )

    print(
        "Checkpoint saved ✅"
    )


if __name__ == "__main__":
    train_model(pretrained=False)