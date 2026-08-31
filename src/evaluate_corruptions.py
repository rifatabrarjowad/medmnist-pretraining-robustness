import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.metrics import balanced_accuracy_score

from medmnistc.utils.baselines import BASELINES


BATCH_SIZE = 64
TEST_SIZE = 624
NUM_SEVERITIES = 5

CORRUPTIONS = [
    "pixelate",
    "jpeg_compression",
    "gaussian_noise",
    "speckle_noise",
    "impulse_noise",
    "shot_noise",
    "gaussian_blur",
    "brightness_up",
    "brightness_down",
    "contrast_up",
    "contrast_down",
    "gamma_corr_up",
    "gamma_corr_down",
]


class CorruptedDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])

        image = self.transform(image)

        label = int(self.labels[idx].squeeze())

        return image, label


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def build_model():
    model = resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        2,
    )

    return model


def load_model(condition, seed, device):
    checkpoint_path = (
        f"results/checkpoints/"
        f"pneumonia_{condition}_seed{seed}.pt"
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint missing: {checkpoint_path}"
        )

    model = build_model()

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model


def evaluate_arrays(
    model,
    images,
    labels,
    device,
):
    dataset = CorruptedDataset(
        images,
        labels,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_images, batch_labels in loader:

            batch_images = batch_images.to(device)

            outputs = model(batch_images)

            preds = torch.argmax(
                outputs,
                dim=1,
            )

            y_true.extend(
                batch_labels.numpy()
            )

            y_pred.extend(
                preds.cpu().numpy()
            )

    return balanced_accuracy_score(
        y_true,
        y_pred,
    )


def verify_npz_structure(images, labels):
    expected = (
        TEST_SIZE
        * NUM_SEVERITIES
    )

    assert len(images) == expected, (
        f"Expected {expected} images, "
        f"found {len(images)}"
    )

    assert len(labels) == expected, (
        f"Expected {expected} labels, "
        f"found {len(labels)}"
    )

    # Verify labels repeat identically for
    # severity blocks 1 through 5.

    reference = labels[:TEST_SIZE]

    for severity in range(NUM_SEVERITIES):

        start = severity * TEST_SIZE
        end = start + TEST_SIZE

        block = labels[start:end]

        assert np.array_equal(
            reference,
            block,
        ), (
            "Label ordering does not match "
            "severity-major assumption."
        )


def get_clean_results():
    path = (
        "results/"
        "pneumonia_clean_test.csv"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing clean results: {path}"
        )

    return pd.read_csv(path)


def main():
    device = get_device()

    print("Device:", device)

    clean_df = get_clean_results()

    baseline = BASELINES[
        "pneumoniamnist"
    ]

    alexnet_clean_error = baseline[
        "clean_score"
    ]

    rows = []

    for corruption in CORRUPTIONS:

        path = (
            "data/corrupted/"
            "pneumoniamnist/"
            f"{corruption}.npz"
        )

        print("\n================================")
        print("Corruption:", corruption)
        print("================================")

        data = np.load(path)

        images = data["test_images"]
        labels = data["test_labels"]

        verify_npz_structure(
            images,
            labels,
        )

        print(
            "NPZ ordering verified ✅"
        )

        alexnet_corrupted_error = (
            baseline["raw_scores"][
                corruption
            ]
        )

        rbe_denominator = (
            alexnet_corrupted_error
            - alexnet_clean_error
        )

        for condition in [
            "pretrained",
            "scratch",
        ]:

            for seed in range(5):

                model = load_model(
                    condition,
                    seed,
                    device,
                )

                clean_row = clean_df[
                    (
                        clean_df["condition"]
                        == condition
                    )
                    &
                    (
                        clean_df["seed"]
                        == seed
                    )
                ]

                clean_bacc = float(
                    clean_row[
                        "clean_test_bacc"
                    ].iloc[0]
                )

                clean_error = (
                    1.0
                    - clean_bacc
                )

                severity_errors = []

                for severity in range(
                    1,
                    NUM_SEVERITIES + 1,
                ):

                    start = (
                        severity - 1
                    ) * TEST_SIZE

                    end = (
                        start
                        + TEST_SIZE
                    )

                    severity_images = (
                        images[start:end]
                    )

                    severity_labels = (
                        labels[start:end]
                    )

                    bacc = evaluate_arrays(
                        model,
                        severity_images,
                        severity_labels,
                        device,
                    )

                    error = 1.0 - bacc

                    severity_errors.append(
                        error
                    )

                    rows.append({
                        "dataset":
                            "pneumonia",
                        "condition":
                            condition,
                        "seed":
                            seed,
                        "corruption":
                            corruption,
                        "severity":
                            severity,
                        "clean_bacc":
                            clean_bacc,
                        "corrupted_bacc":
                            bacc,
                        "corrupted_error":
                            error,
                    })

                    print(
                        f"{condition:10s} "
                        f"seed {seed} "
                        f"severity {severity} "
                        f"| bACC {bacc:.4f}"
                    )

                mean_corrupted_error = (
                    np.mean(
                        severity_errors
                    )
                )

                be = (
                    mean_corrupted_error
                    / alexnet_corrupted_error
                )

                # rBE guard from preregistration
                if rbe_denominator <= 0:
                    rbe = np.nan
                    rbe_valid = False
                else:
                    rbe = (
                        (
                            mean_corrupted_error
                            - clean_error
                        )
                        /
                        rbe_denominator
                    )

                    rbe_valid = True

                for row in rows:

                    if (
                        row["condition"]
                        == condition
                        and row["seed"]
                        == seed
                        and row["corruption"]
                        == corruption
                        and "BE" not in row
                    ):

                        row["BE"] = be
                        row["rBE"] = rbe
                        row[
                            "rBE_valid"
                        ] = rbe_valid

                        row[
                            "alexnet_error"
                        ] = (
                            alexnet_corrupted_error
                        )

                        row[
                            "rBE_denominator"
                        ] = (
                            rbe_denominator
                        )

                del model

                if device.type == "mps":
                    torch.mps.empty_cache()

    df = pd.DataFrame(rows)

    os.makedirs(
        "results",
        exist_ok=True,
    )

    output_path = (
        "results/"
        "pneumonia_corruption_results.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print("\n================================")
    print("EVALUATION COMPLETE ✅")
    print("================================")

    print(
        "Rows:",
        len(df),
    )

    print(
        "Saved:",
        output_path,
    )

    print(
        "\nInvalid rBE corruptions:"
    )

    invalid = (
        df[
            df["rBE_valid"] == False
        ][
            [
                "corruption",
                "rBE_denominator",
            ]
        ]
        .drop_duplicates()
    )

    print(
        invalid.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
