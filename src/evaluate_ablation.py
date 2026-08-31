import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from medmnist import BloodMNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.metrics import balanced_accuracy_score


DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

BATCH_SIZE = 64
NUM_CLASSES = 8

FRACTIONS = [
    1.0,
    0.50,
    0.25,
    0.10,
]

CONDITIONS = [
    "pretrained",
    "scratch",
]

SEEDS = [0, 1, 2]


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


test_dataset = BloodMNIST(
    split="test",
    transform=transform,
    download=True,
    size=224,
    as_rgb=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


def build_model():
    model = resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES,
    )

    return model


def checkpoint_path(
    condition,
    seed,
    fraction,
):
    if fraction == 1.0:
        return (
            f"results/checkpoints/"
            f"blood_{condition}_seed{seed}.pt"
        )

    frac_tag = int(
        fraction * 100
    )

    return (
        f"results/checkpoints/"
        f"blood_{condition}_"
        f"frac{frac_tag}_seed{seed}.pt"
    )


def evaluate(model):
    y_true = []
    y_pred = []

    model.eval()

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            preds = torch.argmax(
                outputs,
                dim=1,
            )

            y_true.extend(
                labels.squeeze(1).numpy()
            )

            y_pred.extend(
                preds.cpu().numpy()
            )

    return balanced_accuracy_score(
        y_true,
        y_pred,
    )


rows = []

print("Device:", DEVICE)

for fraction in FRACTIONS:

    print(
        f"\n============================"
    )
    print(
        f"Training fraction: {fraction}"
    )
    print(
        f"============================"
    )

    for condition in CONDITIONS:

        for seed in SEEDS:

            path = checkpoint_path(
                condition,
                seed,
                fraction,
            )

            if not os.path.exists(path):
                raise FileNotFoundError(path)

            model = build_model()

            state_dict = torch.load(
                path,
                map_location=DEVICE,
                weights_only=True,
            )

            model.load_state_dict(
                state_dict
            )

            model.to(DEVICE)

            bacc = evaluate(model)

            rows.append({
                "fraction": fraction,
                "condition": condition,
                "seed": seed,
                "test_bacc": bacc,
            })

            print(
                f"{condition:10s} "
                f"seed {seed} "
                f"| bACC {bacc:.4f}"
            )

            del model

            if DEVICE.type == "mps":
                torch.mps.empty_cache()


df = pd.DataFrame(rows)

df.to_csv(
    "results/blood_ablation_clean_test.csv",
    index=False,
)


print("\n================================")
print("H2 SUMMARY")
print("================================")

summary = (
    df.groupby(
        [
            "fraction",
            "condition",
        ]
    )["test_bacc"]
    .agg(["mean", "std"])
    .reset_index()
)

print(
    summary.to_string(
        index=False
    )
)


print("\n================================")
print("PRETRAINING GAP")
print("pretrained - scratch")
print("================================")

for fraction in FRACTIONS:

    temp = df[
        df["fraction"] == fraction
    ]

    pretrained = (
        temp[
            temp["condition"]
            == "pretrained"
        ]
        .groupby("seed")[
            "test_bacc"
        ]
        .mean()
    )

    scratch = (
        temp[
            temp["condition"]
            == "scratch"
        ]
        .groupby("seed")[
            "test_bacc"
        ]
        .mean()
    )

    gaps = (
        pretrained
        - scratch
    )

    print(
        f"{int(fraction * 100):3d}% "
        f"| gap mean = "
        f"{gaps.mean():+.4f} "
        f"| SD = {gaps.std():.4f}"
    )


print(
    "\nSaved: "
    "results/blood_ablation_clean_test.csv"
)


# ================================================
# BOOTSTRAP 95% CI FOR PAIRED PRETRAINING GAP
# ================================================

rng = np.random.default_rng(42)
N_BOOT = 10_000

print("\n================================")
print("H2 BOOTSTRAP 95% CI")
print("pretrained - scratch")
print("================================")

ci_rows = []

for fraction in FRACTIONS:

    temp = df[
        df["fraction"] == fraction
    ]

    pre = (
        temp[temp["condition"] == "pretrained"]
        .sort_values("seed")["test_bacc"]
        .to_numpy()
    )

    scr = (
        temp[temp["condition"] == "scratch"]
        .sort_values("seed")["test_bacc"]
        .to_numpy()
    )

    gaps = pre - scr

    boot_means = []

    for _ in range(N_BOOT):
        idx = rng.integers(
            0,
            len(gaps),
            size=len(gaps),
        )

        boot_means.append(
            gaps[idx].mean()
        )

    low, high = np.percentile(
        boot_means,
        [2.5, 97.5],
    )

    mean_gap = gaps.mean()

    ci_rows.append({
        "fraction": fraction,
        "mean_gap": mean_gap,
        "ci_low": low,
        "ci_high": high,
    })

    print(
        f"{int(fraction*100):3d}% "
        f"| gap {mean_gap:+.4f} "
        f"| 95% CI "
        f"[{low:+.4f}, {high:+.4f}]"
    )


pd.DataFrame(ci_rows).to_csv(
    "results/blood_ablation_gap_ci.csv",
    index=False,
)

print(
    "\nSaved: "
    "results/blood_ablation_gap_ci.csv"
)