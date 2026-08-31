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


DATASET_NAME = "derma"
MEDMNISTC_NAME = "dermamnist"

NUM_CLASSES = 7
TEST_SIZE = 2005
NUM_SEVERITIES = 5
BATCH_SIZE = 64

CORRUPTIONS = [
    "pixelate",
    "jpeg_compression",
    "gaussian_noise",
    "speckle_noise",
    "impulse_noise",
    "shot_noise",
    "defocus_blur",
    "motion_blur",
    "zoom_blur",
    "brightness_up",
    "brightness_down",
    "contrast_up",
    "contrast_down",
    "black_corner",
    "characters",
]


class CorruptedDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(
            self.images[idx]
        ).convert("RGB")

        image = self.transform(image)

        label = int(
            self.labels[idx].squeeze()
        )

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
        NUM_CLASSES,
    )

    return model


def load_model(condition, seed, device):
    path = (
        f"results/checkpoints/"
        f"derma_{condition}_seed{seed}.pt"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    model = build_model()

    state_dict = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


def predict_all(
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

            batch_images = batch_images.to(
                device
            )

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

    return (
        np.asarray(y_true),
        np.asarray(y_pred),
    )


def verify_structure(images, labels):
    expected = (
        TEST_SIZE
        * NUM_SEVERITIES
    )

    assert images.shape[0] == expected, (
        f"Expected {expected} images, "
        f"found {images.shape[0]}"
    )

    assert labels.shape[0] == expected, (
        f"Expected {expected} labels, "
        f"found {labels.shape[0]}"
    )

    assert images.shape[-1] == 3, (
        "DermaMNIST-C should be RGB."
    )

    reference = labels[:TEST_SIZE]

    for severity in range(
        NUM_SEVERITIES
    ):
        start = (
            severity
            * TEST_SIZE
        )

        end = (
            start
            + TEST_SIZE
        )

        assert np.array_equal(
            reference,
            labels[start:end],
        ), (
            "Label ordering mismatch at "
            f"severity {severity + 1}"
        )


def main():
    device = get_device()

    print("Device:", device)
    print("Dataset: DermaMNIST-C")

    clean_df = pd.read_csv(
        "results/derma_clean_test.csv"
    )

    baseline = BASELINES[
        MEDMNISTC_NAME
    ]

    alexnet_clean_error = baseline[
        "clean_score"
    ]

    print(
        "AlexNet clean error:",
        alexnet_clean_error,
    )

    rows = []

    for corruption in CORRUPTIONS:

        print("\n================================")
        print("Corruption:", corruption)
        print("================================")

        path = (
            "data/corrupted/"
            "dermamnist/"
            f"{corruption}.npz"
        )

        if not os.path.exists(path):
            raise FileNotFoundError(
                path
            )

        data = np.load(path)

        images = data["test_images"]
        labels = data["test_labels"]

        verify_structure(
            images,
            labels,
        )

        print(
            "NPZ structure verified ✅"
        )

        alexnet_corruption_error = (
            baseline["raw_scores"][
                corruption
            ]
        )

        rbe_denominator = (
            alexnet_corruption_error
            - alexnet_clean_error
        )

        small_denominator = (
            abs(rbe_denominator)
            < 0.05
        )

        for condition in [
            "pretrained",
            "scratch",
        ]:

            for seed in range(5):

                print(
                    f"\n{condition} "
                    f"seed {seed}"
                )

                model = load_model(
                    condition,
                    seed,
                    device,
                )

                y_true, y_pred = (
                    predict_all(
                        model,
                        images,
                        labels,
                        device,
                    )
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

                severity_baccs = []
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

                    bacc = (
                        balanced_accuracy_score(
                            y_true[start:end],
                            y_pred[start:end],
                        )
                    )

                    error = (
                        1.0 - bacc
                    )

                    severity_baccs.append(
                        bacc
                    )

                    severity_errors.append(
                        error
                    )

                    print(
                        f"Severity {severity} "
                        f"| bACC {bacc:.4f}"
                    )

                mean_error = float(
                    np.mean(
                        severity_errors
                    )
                )

                be = (
                    mean_error
                    / alexnet_corruption_error
                )

                if rbe_denominator <= 0:
                    rbe = np.nan
                    rbe_valid = False

                else:
                    rbe = (
                        mean_error
                        - clean_error
                    ) / rbe_denominator

                    rbe_valid = True

                for severity in range(
                    1,
                    NUM_SEVERITIES + 1,
                ):

                    rows.append({
                        "dataset":
                            DATASET_NAME,

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
                            severity_baccs[
                                severity - 1
                            ],

                        "corrupted_error":
                            severity_errors[
                                severity - 1
                            ],

                        "BE":
                            be,

                        "rBE":
                            rbe,

                        "rBE_valid":
                            rbe_valid,

                        "rBE_small_denominator":
                            small_denominator,

                        "alexnet_error":
                            alexnet_corruption_error,

                        "rBE_denominator":
                            rbe_denominator,
                    })

                del model

                if device.type == "mps":
                    torch.mps.empty_cache()

        del images
        del labels
        data.close()

    df = pd.DataFrame(rows)

    output = (
        "results/"
        "derma_corruption_results.csv"
    )

    df.to_csv(
        output,
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
        output,
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
        "\nInvalid rBE corruptions:"
    )

    if len(invalid) == 0:
        print("None")
    else:
        print(
            invalid.to_string(
                index=False
            )
        )

    small = (
        df[
            df[
                "rBE_small_denominator"
            ]
        ][
            [
                "corruption",
                "rBE_denominator",
            ]
        ]
        .drop_duplicates()
    )

    print(
        "\nSmall rBE denominators (< 0.05):"
    )

    if len(small) == 0:
        print("None")
    else:
        print(
            small.to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
