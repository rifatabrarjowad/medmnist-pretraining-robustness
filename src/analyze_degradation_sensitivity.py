import pandas as pd


DATASETS = {
    "blood": {
        "corruption_file": "results/blood_corruption_results.csv",
        "chance": 1 / 8,
    },
    "derma": {
        "corruption_file": "results/derma_corruption_results.csv",
        "chance": 1 / 7,
    },
    "pneumonia": {
        "corruption_file": "results/pneumonia_corruption_results.csv",
        "chance": 1 / 2,
    },
}


rows = []

for dataset, config in DATASETS.items():

    df = pd.read_csv(
        config["corruption_file"]
    )

    chance = config["chance"]

    # Average over all corruption types and severities
    # separately for each condition × seed.
    seed_summary = (
        df.groupby(
            [
                "dataset",
                "condition",
                "seed",
            ],
            as_index=False,
        )
        .agg(
            clean_bacc=(
                "clean_bacc",
                "first",
            ),
            mean_corrupted_bacc=(
                "corrupted_bacc",
                "mean",
            ),
        )
    )

    seed_summary[
        "absolute_drop"
    ] = (
        seed_summary["clean_bacc"]
        - seed_summary["mean_corrupted_bacc"]
    )

    seed_summary[
        "retained_fraction"
    ] = (
        (
            seed_summary["mean_corrupted_bacc"]
            - chance
        )
        /
        (
            seed_summary["clean_bacc"]
            - chance
        )
    )

    seed_summary[
        "chance_normalized_drop"
    ] = (
        1.0
        - seed_summary["retained_fraction"]
    )

    seed_summary["chance_bacc"] = chance

    rows.append(seed_summary)


seed_df = pd.concat(
    rows,
    ignore_index=True,
)

seed_df.to_csv(
    "results/degradation_sensitivity_seed.csv",
    index=False,
)


summary = (
    seed_df.groupby(
        [
            "dataset",
            "condition",
        ],
        as_index=False,
    )
    .agg(
        clean_bacc_mean=(
            "clean_bacc",
            "mean",
        ),
        corrupted_bacc_mean=(
            "mean_corrupted_bacc",
            "mean",
        ),
        absolute_drop_mean=(
            "absolute_drop",
            "mean",
        ),
        retained_fraction_mean=(
            "retained_fraction",
            "mean",
        ),
        chance_normalized_drop_mean=(
            "chance_normalized_drop",
            "mean",
        ),
    )
)

summary.to_csv(
    "results/degradation_sensitivity_summary.csv",
    index=False,
)


wide = summary.pivot(
    index="dataset",
    columns="condition",
    values=[
        "absolute_drop_mean",
        "retained_fraction_mean",
        "chance_normalized_drop_mean",
    ],
)

print("\n========================================")
print("DEGRADATION / FLOOR SENSITIVITY ANALYSIS")
print("========================================")

for dataset in DATASETS:

    sub = summary[
        summary["dataset"] == dataset
    ].set_index("condition")

    print(f"\n{dataset.upper()}")

    for condition in [
        "pretrained",
        "scratch",
    ]:
        row = sub.loc[condition]

        print(
            f"{condition:>10} "
            f"| clean={row['clean_bacc_mean']:.4f} "
            f"| corrupt={row['corrupted_bacc_mean']:.4f} "
            f"| abs_drop={row['absolute_drop_mean']:.4f} "
            f"| retained={row['retained_fraction_mean']:.4f}"
        )

    retained_gap = (
        sub.loc[
            "pretrained",
            "retained_fraction_mean",
        ]
        -
        sub.loc[
            "scratch",
            "retained_fraction_mean",
        ]
    )

    drop_gap = (
        sub.loc[
            "scratch",
            "absolute_drop_mean",
        ]
        -
        sub.loc[
            "pretrained",
            "absolute_drop_mean",
        ]
    )

    print(
        f"Pretrained advantage in retained skill: "
        f"{retained_gap:+.4f}"
    )

    print(
        f"Pretrained advantage in absolute drop "
        f"(positive = smaller drop): "
        f"{drop_gap:+.4f}"
    )


print("\nSaved:")
print("results/degradation_sensitivity_seed.csv")
print("results/degradation_sensitivity_summary.csv")
