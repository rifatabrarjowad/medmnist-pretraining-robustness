import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.metrics import balanced_accuracy_score

from medmnist import BloodMNIST, DermaMNIST, PneumoniaMNIST


DATASETS = {
    "blood": (BloodMNIST, 8),
    "derma": (DermaMNIST, 7),
    "pneumonia": (PneumoniaMNIST, 2),
}

BATCH_SIZE = 64


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


def get_test_loader(dataset_name):
    dataset_class, _ = DATASETS[dataset_name]

    test_dataset = dataset_class(
        split="test",
        transform=get_transform(),
        download=True,
        size=224,
        as_rgb=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return test_loader


def build_model(dataset_name):
    _, num_classes = DATASETS[dataset_name]

    model = resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    return model


def evaluate_checkpoint(
    dataset_name,
    condition,
    seed,
    device,
):
    checkpoint_path = (
        f"results/checkpoints/"
        f"{dataset_name}_{condition}_seed{seed}.pt"
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    model = build_model(dataset_name)

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    test_loader = get_test_loader(dataset_name)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
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

    bacc = balanced_accuracy_score(
        y_true,
        y_pred
    )

    return bacc


def main(dataset_name):
    device = get_device()

    print("Device:", device)
    print("Dataset:", dataset_name)

    rows = []

    for condition in [
        "pretrained",
        "scratch",
    ]:

        for seed in range(5):

            bacc = evaluate_checkpoint(
                dataset_name,
                condition,
                seed,
                device,
            )

            print(
                f"{condition:10s} "
                f"seed {seed} "
                f"| Test bACC {bacc:.4f}"
            )

            rows.append({
                "dataset": dataset_name,
                "condition": condition,
                "seed": seed,
                "clean_test_bacc": bacc,
            })

    df = pd.DataFrame(rows)

    os.makedirs(
        "results",
        exist_ok=True
    )

    output_path = (
        f"results/"
        f"{dataset_name}_clean_test.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print("\n========================")
    print("SUMMARY")
    print("========================")

    summary = (
        df.groupby("condition")
        ["clean_test_bacc"]
        .agg(["mean", "std"])
    )

    print(summary)

    print(
        f"\nSaved: {output_path}"
    )


if __name__ == "__main__":
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

    args = parser.parse_args()

    main(args.dataset)
