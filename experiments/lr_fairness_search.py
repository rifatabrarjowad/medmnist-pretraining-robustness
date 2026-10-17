import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)
from medmnist import BloodMNIST
from sklearn.metrics import balanced_accuracy_score


SEED = 0

BATCH_SIZE = 64
MAX_EPOCHS = 50
PATIENCE = 10

CONDITIONS = [
    "pretrained",
    "scratch",
]

LEARNING_RATES = [
    1e-4,
    3e-4,
    1e-3,
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_loaders():
    transform = get_transform()

    train_dataset = BloodMNIST(
        split="train",
        transform=transform,
        download=True,
        size=224,
        as_rgb=True,
    )

    val_dataset = BloodMNIST(
        split="val",
        transform=transform,
        download=True,
        size=224,
        as_rgb=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, val_loader


def build_model(condition):
    if condition == "pretrained":
        model = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1
        )

    elif condition == "scratch":
        model = resnet18(
            weights=None
        )

    else:
        raise ValueError(
            f"Unknown condition: {condition}"
        )

    model.fc = nn.Linear(
        model.fc.in_features,
        8,
    )

    return model


def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            labels = (
                labels
                .squeeze(1)
                .long()
                .to(device)
            )

            outputs = model(images)

            preds = torch.argmax(
                outputs,
                dim=1,
            )

            y_true.extend(
                labels.cpu().numpy()
            )

            y_pred.extend(
                preds.cpu().numpy()
            )

    return balanced_accuracy_score(
        y_true,
        y_pred,
    )


def run_one(
    condition,
    lr,
    device,
):
    # Reset randomness before every
    # condition × LR run.
    set_seed(SEED)

    train_loader, val_loader = get_loaders()

    model = build_model(
        condition
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=MAX_EPOCHS,
        )
    )

    best_val_bacc = -1.0
    best_epoch = -1
    patience_counter = 0

    print("\n================================")
    print("Condition:", condition)
    print("Learning rate:", lr)
    print("Seed:", SEED)
    print("================================")

    for epoch in range(MAX_EPOCHS):
        model.train()

        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)

            labels = (
                labels
                .squeeze(1)
                .long()
                .to(device)
            )

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
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
            device,
        )

        print(
            f"Epoch {epoch + 1:02d}/{MAX_EPOCHS} "
            f"| Loss {avg_loss:.4f} "
            f"| Val bACC {val_bacc:.4f}"
        )

        if val_bacc > best_val_bacc:
            best_val_bacc = val_bacc
            best_epoch = epoch + 1
            patience_counter = 0

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping")
            break

    result = {
        "condition": condition,
        "lr": lr,
        "seed": SEED,
        "best_val_bacc": best_val_bacc,
        "best_epoch": best_epoch,
    }

    del model

    if device.type == "mps":
        torch.mps.empty_cache()

    return result


def main():
    device = get_device()

    print("Device:", device)
    print("Dataset: BloodMNIST")
    print("Seed:", SEED)
    print(
        "LR grid:",
        LEARNING_RATES,
    )
    print(
        "Protocol:",
        f"{MAX_EPOCHS} epochs max, "
        f"patience={PATIENCE}",
    )

    results = []

    for condition in CONDITIONS:
        for lr in LEARNING_RATES:
            result = run_one(
                condition,
                lr,
                device,
            )

            results.append(result)

    df = pd.DataFrame(results)

    output = (
        "results/"
        "lr_fairness_search.csv"
    )

    df.to_csv(
        output,
        index=False,
    )

    print("\n================================")
    print("LR FAIRNESS SEARCH RESULTS")
    print("================================")

    print(
        df.to_string(
            index=False
        )
    )

    print("\n================================")
    print("BEST LR BY CONDITION")
    print("================================")

    for condition in CONDITIONS:
        subset = df[
            df["condition"] == condition
        ]

        best_row = subset.loc[
            subset["best_val_bacc"].idxmax()
        ]

        print(
            f"{condition:>10}: "
            f"LR={best_row['lr']:.4g} "
            f"| best val bACC="
            f"{best_row['best_val_bacc']:.6f} "
            f"| epoch="
            f"{int(best_row['best_epoch'])}"
        )

    print("\nSaved:")
    print(output)


if __name__ == "__main__":
    main()
