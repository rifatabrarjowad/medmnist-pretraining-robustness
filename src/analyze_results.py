import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


FILES = [
    "results/pneumonia_corruption_results.csv",
    "results/blood_corruption_results.csv",
    "results/derma_corruption_results.csv",
]

df = pd.concat(
    [pd.read_csv(f) for f in FILES],
    ignore_index=True,
)


FAMILY_MAP = {
    # Blood
    ("blood", "pixelate"): "digital",
    ("blood", "jpeg_compression"): "digital",
    ("blood", "defocus_blur"): "blur",
    ("blood", "motion_blur"): "blur",
    ("blood", "brightness_up"): "color",
    ("blood", "brightness_down"): "color",
    ("blood", "contrast_up"): "color",
    ("blood", "contrast_down"): "color",
    ("blood", "saturate"): "color",
    ("blood", "bubble"): "task_specific",
    ("blood", "stain_deposit"): "task_specific",

    # Derma
    ("derma", "pixelate"): "digital",
    ("derma", "jpeg_compression"): "digital",
    ("derma", "gaussian_noise"): "noise",
    ("derma", "speckle_noise"): "noise",
    ("derma", "impulse_noise"): "noise",
    ("derma", "shot_noise"): "noise",
    ("derma", "defocus_blur"): "blur",
    ("derma", "motion_blur"): "blur",
    ("derma", "zoom_blur"): "blur",
    ("derma", "brightness_up"): "color",
    ("derma", "brightness_down"): "color",
    ("derma", "contrast_up"): "color",
    ("derma", "contrast_down"): "color",
    ("derma", "black_corner"): "task_specific",
    ("derma", "characters"): "task_specific",

    # Pneumonia
    ("pneumonia", "pixelate"): "digital",
    ("pneumonia", "jpeg_compression"): "digital",
    ("pneumonia", "gaussian_noise"): "noise",
    ("pneumonia", "speckle_noise"): "noise",
    ("pneumonia", "impulse_noise"): "noise",
    ("pneumonia", "shot_noise"): "noise",
    ("pneumonia", "gaussian_blur"): "blur",
    ("pneumonia", "brightness_up"): "color",
    ("pneumonia", "brightness_down"): "color",
    ("pneumonia", "contrast_up"): "color",
    ("pneumonia", "contrast_down"): "color",
    ("pneumonia", "gamma_corr_up"): "color",
    ("pneumonia", "gamma_corr_down"): "color",
}


df["family"] = [
    FAMILY_MAP[(d, c)]
    for d, c in zip(
        df["dataset"],
        df["corruption"],
    )
]


# --------------------------------------------------
# 1. Collapse severity repetitions
#    BE is already corruption-level and repeated
#    across the five severity rows.
# --------------------------------------------------

per_seed_corruption = (
    df.groupby(
        [
            "dataset",
            "condition",
            "seed",
            "corruption",
            "family",
        ]
    )
    .agg(
        corrupted_bacc=("corrupted_bacc", "mean"),
        BE=("BE", "first"),
        clean_bacc=("clean_bacc", "first"),
    )
    .reset_index()
)


# --------------------------------------------------
# 2. Mean over seeds for each corruption
# --------------------------------------------------

per_corruption = (
    per_seed_corruption
    .groupby(
        [
            "dataset",
            "condition",
            "corruption",
            "family",
        ]
    )
    .agg(
        corrupted_bacc_mean=("corrupted_bacc", "mean"),
        corrupted_bacc_sd=("corrupted_bacc", "std"),
        BE_mean=("BE", "mean"),
        BE_sd=("BE", "std"),
    )
    .reset_index()
)


# --------------------------------------------------
# 3. Pretrained - Scratch difference
#
# For bACC:
# positive = pretrained better
#
# For BE:
# lower is better, so define:
# delta_BE = scratch_BE - pretrained_BE
# positive = pretrained better
# --------------------------------------------------

pre = per_corruption[
    per_corruption["condition"] == "pretrained"
].copy()

scr = per_corruption[
    per_corruption["condition"] == "scratch"
].copy()

merged = pre.merge(
    scr,
    on=[
        "dataset",
        "corruption",
        "family",
    ],
    suffixes=("_pre", "_scratch"),
)

merged["delta_bacc"] = (
    merged["corrupted_bacc_mean_pre"]
    - merged["corrupted_bacc_mean_scratch"]
)

merged["delta_BE"] = (
    merged["BE_mean_scratch"]
    - merged["BE_mean_pre"]
)


print("\n====================================")
print("PER-CORRUPTION EFFECT")
print("positive delta = pretrained better")
print("====================================")

for dataset in [
    "blood",
    "derma",
    "pneumonia",
]:
    temp = merged[
        merged["dataset"] == dataset
    ].sort_values(
        "delta_bacc",
        ascending=False,
    )

    print(f"\n--- {dataset.upper()} ---")

    print(
        temp[
            [
                "corruption",
                "family",
                "delta_bacc",
                "delta_BE",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# --------------------------------------------------
# 4. Family summaries
# --------------------------------------------------

family_summary = (
    merged.groupby(
        [
            "dataset",
            "family",
        ]
    )
    .agg(
        n_corruptions=("corruption", "count"),
        mean_delta_bacc=("delta_bacc", "mean"),
        mean_delta_BE=("delta_BE", "mean"),
    )
    .reset_index()
)

print("\n====================================")
print("FAMILY SUMMARY")
print("positive = pretrained better")
print("====================================")

print(
    family_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# --------------------------------------------------
# 5. Wilcoxon within each dataset
# --------------------------------------------------

print("\n====================================")
print("WILCOXON: PER-CORRUPTION DELTA BE")
print("====================================")

for dataset in [
    "blood",
    "derma",
    "pneumonia",
]:
    values = merged.loc[
        merged["dataset"] == dataset,
        "delta_BE",
    ].values

    stat, p = wilcoxon(
        values,
        alternative="two-sided",
    )

    print(
        f"{dataset:10s} "
        f"n={len(values):2d} "
        f"median_delta_BE={np.median(values):.4f} "
        f"W={stat:.3f} "
        f"p={p:.6f}"
    )


# --------------------------------------------------
# 6. Primary endpoint:
# task-specific vs pooled noise/blur/digital
# Blood + Derma only
# --------------------------------------------------

print("\n====================================")
print("PRIMARY ENDPOINT")
print("====================================")

GENERAL = [
    "noise",
    "blur",
    "digital",
]

for dataset in [
    "blood",
    "derma",
]:
    temp = merged[
        merged["dataset"] == dataset
    ]

    task = temp[
        temp["family"]
        == "task_specific"
    ]["delta_BE"]

    general = temp[
        temp["family"].isin(GENERAL)
    ]["delta_BE"]

    task_mean = task.mean()
    general_mean = general.mean()

    interaction = (
        task_mean
        - general_mean
    )

    print(f"\n{dataset.upper()}")
    print(
        "Task-specific mean ΔBE:",
        round(task_mean, 4),
    )
    print(
        "Noise/blur/digital mean ΔBE:",
        round(general_mean, 4),
    )
    print(
        "Interaction Δ:",
        round(interaction, 4),
    )


merged.to_csv(
    "results/per_corruption_effects.csv",
    index=False,
)

family_summary.to_csv(
    "results/family_summary.csv",
    index=False,
)

print("\nSaved:")
print(
    "results/per_corruption_effects.csv"
)
print(
    "results/family_summary.csv"
)


# ==================================================
# 7. Bootstrap 95% confidence intervals
# ==================================================

RNG = np.random.default_rng(42)
N_BOOT = 10_000


def bootstrap_mean_ci(values, n_boot=N_BOOT):
    values = np.asarray(values, dtype=float)

    boot_means = []

    for _ in range(n_boot):
        sample = RNG.choice(
            values,
            size=len(values),
            replace=True,
        )

        boot_means.append(
            np.mean(sample)
        )

    low, high = np.percentile(
        boot_means,
        [2.5, 97.5],
    )

    return (
        float(np.mean(values)),
        float(low),
        float(high),
    )


print("\n====================================")
print("BOOTSTRAP 95% CI: FAMILY ΔBE")
print("positive = pretrained better")
print("====================================")

bootstrap_rows = []

for dataset in [
    "blood",
    "derma",
    "pneumonia",
]:

    temp = merged[
        merged["dataset"] == dataset
    ]

    for family in sorted(
        temp["family"].unique()
    ):

        values = temp.loc[
            temp["family"] == family,
            "delta_BE",
        ].values

        mean, low, high = (
            bootstrap_mean_ci(values)
        )

        bootstrap_rows.append({
            "dataset": dataset,
            "family": family,
            "n": len(values),
            "mean_delta_BE": mean,
            "ci_low": low,
            "ci_high": high,
        })

        print(
            f"{dataset:10s} "
            f"{family:14s} "
            f"n={len(values):2d} "
            f"ΔBE={mean:+.4f} "
            f"95% CI [{low:+.4f}, {high:+.4f}]"
        )


bootstrap_df = pd.DataFrame(
    bootstrap_rows
)


# ==================================================
# 8. Bootstrap primary interaction
#
# interaction =
# mean(task-specific ΔBE)
# -
# mean(noise/blur/digital ΔBE)
# ==================================================

print("\n====================================")
print("BOOTSTRAP PRIMARY INTERACTION")
print("====================================")

interaction_rows = []

for dataset in [
    "blood",
    "derma",
]:

    temp = merged[
        merged["dataset"] == dataset
    ]

    task = temp.loc[
        temp["family"] == "task_specific",
        "delta_BE",
    ].values

    general = temp.loc[
        temp["family"].isin(
            ["noise", "blur", "digital"]
        ),
        "delta_BE",
    ].values

    boot_interactions = []

    for _ in range(N_BOOT):

        task_sample = RNG.choice(
            task,
            size=len(task),
            replace=True,
        )

        general_sample = RNG.choice(
            general,
            size=len(general),
            replace=True,
        )

        interaction = (
            np.mean(task_sample)
            -
            np.mean(general_sample)
        )

        boot_interactions.append(
            interaction
        )

    observed = (
        np.mean(task)
        -
        np.mean(general)
    )

    low, high = np.percentile(
        boot_interactions,
        [2.5, 97.5],
    )

    interaction_rows.append({
        "dataset": dataset,
        "interaction": observed,
        "ci_low": low,
        "ci_high": high,
    })

    print(
        f"{dataset:10s} "
        f"interaction={observed:+.4f} "
        f"95% CI [{low:+.4f}, {high:+.4f}]"
    )


interaction_df = pd.DataFrame(
    interaction_rows
)


# ==================================================
# 9. Family-level Wilcoxon + Holm correction
# ==================================================

def holm_adjust(p_values):
    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    n = len(p_values)

    order = np.argsort(p_values)

    adjusted = np.empty(
        n,
        dtype=float,
    )

    running_max = 0.0

    for rank, idx in enumerate(order):

        multiplier = n - rank

        value = (
            multiplier
            * p_values[idx]
        )

        value = min(
            value,
            1.0,
        )

        running_max = max(
            running_max,
            value,
        )

        adjusted[idx] = running_max

    return adjusted


print("\n====================================")
print("FAMILY WILCOXON + HOLM")
print("====================================")

holm_rows = []

for dataset in [
    "blood",
    "derma",
    "pneumonia",
]:

    temp = merged[
        merged["dataset"] == dataset
    ]

    tests = []

    for family in sorted(
        temp["family"].unique()
    ):

        values = temp.loc[
            temp["family"] == family,
            "delta_BE",
        ].values

        # Wilcoxon requires enough paired
        # corruption observations to be useful.
        if len(values) < 2:
            continue

        try:
            stat, p = wilcoxon(
                values,
                alternative="two-sided",
            )

        except ValueError:
            stat = np.nan
            p = 1.0

        tests.append({
            "dataset": dataset,
            "family": family,
            "n": len(values),
            "median_delta_BE":
                float(np.median(values)),
            "W": stat,
            "p_raw": p,
        })

    if len(tests) == 0:
        continue

    raw_p = [
        x["p_raw"]
        for x in tests
    ]

    corrected = holm_adjust(
        raw_p
    )

    for row, p_holm in zip(
        tests,
        corrected,
    ):
        row["p_holm"] = p_holm

        holm_rows.append(row)

        print(
            f'{row["dataset"]:10s} '
            f'{row["family"]:14s} '
            f'n={row["n"]:2d} '
            f'median={row["median_delta_BE"]:+.4f} '
            f'p={row["p_raw"]:.6f} '
            f'p_holm={p_holm:.6f}'
        )


holm_df = pd.DataFrame(
    holm_rows
)


# ==================================================
# SAVE
# ==================================================

bootstrap_df.to_csv(
    "results/bootstrap_family_ci.csv",
    index=False,
)

interaction_df.to_csv(
    "results/bootstrap_primary_interaction.csv",
    index=False,
)

holm_df.to_csv(
    "results/family_wilcoxon_holm.csv",
    index=False,
)

print("\nSaved:")
print("results/bootstrap_family_ci.csv")
print("results/bootstrap_primary_interaction.csv")
print("results/family_wilcoxon_holm.csv")