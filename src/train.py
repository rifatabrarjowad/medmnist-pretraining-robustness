import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import balanced_accuracy_score

from medmnist import BloodMNIST, DermaMNIST, PneumoniaMNIST


DATASETS = {
    "blood": (BloodMNIST, 8),
    "derma": (DermaMNIST, 7),
    "pneumonia": (PneumoniaMNIST, 2),
}

PRETRAINED_LR = 1e-4
SCRATCH_LR = 3e-4

BATCH_SIZE = 64
MAX_EPOCHS = 50
PATIENCE = 10


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


def get_dataloaders(dataset_name, fraction=1.0, seed=0):
    dataset_class, _ = DATASETS[dataset_name]

    transform = get_transform()

    train_dataset = dataset_class(
        split="train",
        transform=transform,
        download=True,
        size=224,
        as_rgb=True,
    )
    if fraction < 1.0:
        generator = torch.Generator().manual_seed(seed)

        subset_size = int(
            len(train_dataset) * fraction
        )

        indices = torch.randperm(
            len(train_dataset),
            generator=generator,
        )[:subset_size]

        train_dataset = torch.utils.data.Subset(
            train_dataset,
            indices.tolist(),
        )

        print(
            f"Training fraction: {fraction}"
        )
        print(
            f"Training samples: {len(train_dataset)}"
        )

    val_dataset = dataset_class(
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


def build_model(dataset_name, condition):
    _, num_classes = DATASETS[dataset_name]

    if condition == "pretrained":
        model = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1
        )
    else:
        model = resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
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

            preds = torch.argmax(
                outputs,
                dim=1
            )

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


def train(
    dataset_name,
    condition,
    seed,
    fraction=1.0,
):
    set_seed(seed)

    device = get_device()

    print("Device:", device)
    print("Dataset:", dataset_name)
    print("Condition:", condition)
    print("Seed:", seed)

    train_loader, val_loader = get_dataloaders(
    dataset_name,
    fraction=fraction,
    seed=seed,
    )

    model = build_model(
        dataset_name,
        condition
    ).to(device)

    if condition == "pretrained":
        lr = PRETRAINED_LR
    else:
        lr = SCRATCH_LR

    print("Learning rate:", lr)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS,
    )

    best_val_bacc = -1.0
    best_epoch = -1
    patience_counter = 0

    os.makedirs(
        "results/checkpoints",
        exist_ok=True
    )

    if fraction == 1.0:
        checkpoint_path = (
            f"results/checkpoints/"
            f"{dataset_name}_{condition}_seed{seed}.pt"
        )
    else:
        frac_tag = int(fraction * 100)

        checkpoint_path = (
            f"results/checkpoints/"
            f"{dataset_name}_{condition}_"
            f"frac{frac_tag}_seed{seed}.pt"
        )

    for epoch in range(MAX_EPOCHS):
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
            running_loss /
            len(train_loader)
        )

        val_bacc = evaluate(
            model,
            val_loader,
            device
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

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping")
            break

    print("\nTraining complete")
    print("Best epoch:", best_epoch)
    print("Best val bACC:", best_val_bacc)
    print("Saved:", checkpoint_path)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "blood",
            "derma",
            "pneumonia",
        ],
    )

    parser.add_argument(
        "--condition",
        required=True,
        choices=[
            "pretrained",
            "scratch",
        ],
    )

    parser.add_argument(
        "--seed",
        required=True,
        type=int,
    )
    parser.add_argument(
    "--fraction",
    type=float,
    default=1.0,
    choices=[
        1.0,
        0.5,
        0.25,
        0.10,
    ],
)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train(
    dataset_name=args.dataset,
    condition=args.condition,
    seed=args.seed,
    fraction=args.fraction,
)