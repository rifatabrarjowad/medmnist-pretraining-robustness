from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

N_BOOT = 10_000
BOOT_SEED = 2026

FILES = {
    "blood": "results/blood_corruption_results.csv",
    "derma": "results/derma_corruption_results.csv",
    "pneumonia": "results/pneumonia_corruption_results.csv",
}

# Same family definitions used in the previous analyses.
FAMILY_MAP = {
    # Generic digital
    "pixelate": "digital",
    "jpeg_compression": "digital",

    # Blur
    "defocus_blur": "blur",
    "motion_blur": "blur",
    "zoom_blur": "blur",
    "gaussian_blur": "blur",

    # Noise
    "gaussian_noise": "noise",
    "shot_noise": "noise",
    "impulse_noise": "noise",
    "speckle_noise": "noise",

    # Color / intensity
    "brightness_down": "color",
    "brightness_up": "color",
    "contrast_down": "color",
    "contrast_up": "color",
    "saturate": "color",
    "gamma_corr_down": "color",
    "gamma_corr_up": "color",

    # Task-specific
    "stain_deposit": "task",
    "bubble": "task",
    "black_corner": "task",
    "characters": "task",
}


def bootstrap_mean_ci(values, n_boot=N_BOOT, seed=BOOT_SEED):
    """
    Paired-seed bootstrap.

    `values` is already the pretrained-minus-scratch gap
    for each matched seed.
    """
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    samples = rng.choice(
        values,
        size=(n_boot, len(values)),
        replace=True,
    )

    means = samples.mean(axis=1)

    low, high = np.percentile(
        means,
        [2.5, 97.5],
    )

    return low, high


all_rows = []

for dataset, filename in FILES.items():

    df = pd.read_csv(filename)

    df["family"] = df["corruption"].map(FAMILY_MAP)

    missing = sorted(
        df.loc[df["family"].isna(), "corruption"].unique()
    )

    if missing:
        raise ValueError(
            f"{dataset}: missing family assignments: {missing}"
        )

    # --------------------------------------------------------
    # First average corruptions WITHIN each family for each
    # condition × seed × severity.
    #
    # This preserves seed as the uncertainty unit.
    # --------------------------------------------------------

    seed_family = (
        df.groupby(
            [
                "dataset",
                "condition",
                "seed",
                "family",
                "severity",
            ],
            as_index=False,
        )["corrupted_bacc"]
        .mean()
    )

    # Put pretrained and scratch results on the same row.
    paired = seed_family.pivot(
        index=[
            "dataset",
            "seed",
            "family",
            "severity",
        ],
        columns="condition",
        values="corrupted_bacc",
    ).reset_index()

    if "pretrained" not in paired.columns or "scratch" not in paired.columns:
        raise ValueError(
            f"{dataset}: pretrained/scratch condition missing."
        )

    # Positive = pretrained better.
    paired["gap"] = (
        paired["pretrained"]
        - paired["scratch"]
    )

    # --------------------------------------------------------
    # Bootstrap matched seed gaps
    # --------------------------------------------------------

    for (
        dataset_name,
        family,
        severity,
    ), group in paired.groupby(
        ["dataset", "family", "severity"]
    ):

        gaps = group["gap"].to_numpy()

        ci_low, ci_high = bootstrap_mean_ci(
            gaps,
            seed=(
                BOOT_SEED
                + int(severity)
                + sum(ord(c) for c in dataset_name + family)
            ),
        )

        all_rows.append(
            {
                "dataset": dataset_name,
                "family": family,
                "severity": int(severity),
                "n_seeds": len(gaps),
                "mean_gap": gaps.mean(),
                "sd_gap": gaps.std(ddof=1)
                if len(gaps) > 1
                else np.nan,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "monotonic_metric": "pretrained_minus_scratch_bacc",
            }
        )


out = pd.DataFrame(all_rows).sort_values(
    ["dataset", "family", "severity"]
)

Path("results").mkdir(exist_ok=True)

out.to_csv(
    "results/h4_severity_family_ci.csv",
    index=False,
)

print("\nSaved:")
print("results/h4_severity_family_ci.csv")

print("\nPreview:")
print(
    out.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)
