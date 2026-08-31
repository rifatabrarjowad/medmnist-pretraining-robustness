import pandas as pd
import numpy as np


FILES = [
    "results/blood_corruption_results.csv",
    "results/derma_corruption_results.csv",
    "results/pneumonia_corruption_results.csv",
]

df = pd.concat(
    [pd.read_csv(f) for f in FILES],
    ignore_index=True,
)


FAMILY_MAP = {
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


# Mean bACC over corruptions and seeds
severity = (
    df.groupby(
        [
            "dataset",
            "condition",
            "family",
            "severity",
        ]
    )["corrupted_bacc"]
    .mean()
    .reset_index()
)


pre = severity[
    severity["condition"] == "pretrained"
]

scr = severity[
    severity["condition"] == "scratch"
]


merged = pre.merge(
    scr,
    on=[
        "dataset",
        "family",
        "severity",
    ],
    suffixes=("_pre", "_scratch"),
)


merged["gap"] = (
    merged["corrupted_bacc_pre"]
    - merged["corrupted_bacc_scratch"]
)


print("\n====================================")
print("H4 SEVERITY GAP")
print("positive = pretrained better")
print("====================================")


for dataset in [
    "blood",
    "derma",
    "pneumonia",
]:

    print(f"\n--- {dataset.upper()} ---")

    temp = merged[
        merged["dataset"] == dataset
    ]

    for family in sorted(
        temp["family"].unique()
    ):

        fam = (
            temp[temp["family"] == family]
            .sort_values("severity")
        )

        gaps = fam["gap"].to_numpy()

        monotonic_decreasing = np.all(
            np.diff(gaps) <= 0
        )

        values = " → ".join(
            f"{x:+.4f}"
            for x in gaps
        )

        print(
            f"{family:14s} "
            f"{values} "
            f"| decreasing={monotonic_decreasing}"
        )


# Overall dataset-level severity gap
overall = (
    df.groupby(
        [
            "dataset",
            "condition",
            "severity",
        ]
    )["corrupted_bacc"]
    .mean()
    .reset_index()
)


pre_o = overall[
    overall["condition"] == "pretrained"
]

scr_o = overall[
    overall["condition"] == "scratch"
]


overall_gap = pre_o.merge(
    scr_o,
    on=[
        "dataset",
        "severity",
    ],
    suffixes=("_pre", "_scratch"),
)


overall_gap["gap"] = (
    overall_gap["corrupted_bacc_pre"]
    - overall_gap["corrupted_bacc_scratch"]
)


print("\n====================================")
print("OVERALL DATASET SEVERITY GAP")
print("====================================")


for dataset in [
    "blood",
    "derma",
    "pneumonia",
]:

    temp = (
        overall_gap[
            overall_gap["dataset"] == dataset
        ]
        .sort_values("severity")
    )

    gaps = temp["gap"].to_numpy()

    print(
        f"{dataset:10s}: "
        + " → ".join(
            f"{x:+.4f}"
            for x in gaps
        )
    )


# Count dataset × family cells
# satisfying preregistered monotonic rule

rows = []

for (
    dataset,
    family
), temp in merged.groupby(
    ["dataset", "family"]
):

    temp = temp.sort_values(
        "severity"
    )

    gaps = temp["gap"].to_numpy()

    decreasing = bool(
        np.all(
            np.diff(gaps) <= 0
        )
    )

    rows.append({
        "dataset": dataset,
        "family": family,
        "monotonic_decreasing": decreasing,
        "severity1_gap": gaps[0],
        "severity5_gap": gaps[-1],
    })


trend_df = pd.DataFrame(rows)

print("\n====================================")
print("H4 DECISION COUNT")
print("====================================")

n_total = len(trend_df)

n_decreasing = int(
    trend_df["monotonic_decreasing"].sum()
)

print(
    f"Monotonic decreasing cells: "
    f"{n_decreasing}/{n_total}"
)

print(
    f"Fraction: "
    f"{n_decreasing / n_total:.3f}"
)

print("\nCells:")
print(
    trend_df.to_string(
        index=False
    )
)


merged.to_csv(
    "results/h4_severity_family.csv",
    index=False,
)

trend_df.to_csv(
    "results/h4_trend_summary.csv",
    index=False,
)

print("\nSaved:")
print("results/h4_severity_family.csv")
print("results/h4_trend_summary.csv")
