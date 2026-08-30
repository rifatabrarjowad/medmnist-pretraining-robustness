import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from medmnist import BloodMNIST
from sklearn.metrics import balanced_accuracy_score


SEED = 0
BATCH_SIZE = 64
MAX_EPOCHS = 20
PATIENCE = 5

LEARNING_RATES = [
    1e-4,
    3e-4,
    1e-3
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


def get_loaders():
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = BloodMNIST(
        split="train",
        transform=transform,
        download=True,
        size=224
    )

    val_dataset = BloodMNIST(
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


def build_scratch_model():
    model = resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        8
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


def run_lr_search(lr, train_loader, val_loader, device):
    set_seed(SEED)

    model = build_scratch_model().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS
    )

    best_val_bacc = -1.0
    best_epoch = -1
    patience_counter = 0

    print(f"\nTesting LR = {lr}")

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
            running_loss
            / len(train_loader)
        )

        val_bacc = evaluate(
            model,
            val_loader,
            device
        )

        print(
            f"Epoch {epoch + 1:02d} "
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

    return {
        "lr": lr,
        "best_val_bacc": best_val_bacc,
        "best_epoch": best_epoch
    }


def main():
    device = get_device()

    print("Device:", device)

    train_loader, val_loader = get_loaders()

    print("BloodMNIST loaded")
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))

    results = []

    for lr in LEARNING_RATES:
        result = run_lr_search(
            lr,
            train_loader,
            val_loader,
            device
        )

        results.append(result)

    df = pd.DataFrame(results)

    print("\n==============================")
    print("LEARNING RATE SEARCH RESULTS")
    print("==============================")

    print(df.to_string(index=False))

    best_row = df.loc[
        df["best_val_bacc"].idxmax()
    ]

    selected_lr = best_row["lr"]

    print("\nSelected scratch LR:")
    print(selected_lr)

    df.to_csv(
        "results/lr_search.csv",
        index=False
    )

    print(
        "\nSaved to results/lr_search.csv"
    )


if __name__ == "__main__":
    main()
