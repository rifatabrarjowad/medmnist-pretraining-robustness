import os
import pandas as pd
import matplotlib.pyplot as plt


os.makedirs("figures", exist_ok=True)


# ============================================================
# FIGURE 1 — FAMILY-LEVEL PRETRAINING EFFECT
# ============================================================

family = pd.read_csv(
    "results/family_summary.csv"
)

datasets = [
    "blood",
    "derma",
    "pneumonia",
]

for dataset in datasets:

    temp = family[
        family["dataset"] == dataset
    ].copy()

    temp = temp.sort_values(
        "mean_delta_bacc"
    )

    plt.figure(figsize=(8, 5))

    plt.barh(
        temp["family"],
        temp["mean_delta_bacc"] * 100,
    )

    plt.axvline(
        0,
        linewidth=1,
    )

    plt.xlabel(
        "Pretrained − Scratch balanced accuracy (percentage points)"
    )

    plt.ylabel(
        "Corruption family"
    )

    plt.title(
        f"{dataset.capitalize()}MNIST: effect of ImageNet pretraining"
    )

    plt.tight_layout()

    plt.savefig(
        f"figures/{dataset}_family_effect.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        f"figures/{dataset}_family_effect.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# FIGURE 2 — H2 TRAINING FRACTION
# ============================================================

h2 = pd.read_csv(
    "results/blood_ablation_gap_ci.csv"
)

h2 = h2.sort_values(
    "fraction"
)

x = h2["fraction"] * 100
y = h2["mean_gap"] * 100

lower = (
    h2["mean_gap"]
    - h2["ci_low"]
) * 100

upper = (
    h2["ci_high"]
    - h2["mean_gap"]
) * 100


plt.figure(figsize=(7, 5))

plt.errorbar(
    x,
    y,
    yerr=[lower, upper],
    marker="o",
    capsize=5,
)

plt.axhline(
    0,
    linewidth=1,
)

plt.xlabel(
    "Training data used (%)"
)

plt.ylabel(
    "Pretrained − Scratch bACC (percentage points)"
)

plt.title(
    "BloodMNIST: pretraining advantage vs training-set size"
)

plt.xticks(
    [10, 25, 50, 100]
)

plt.tight_layout()

plt.savefig(
    "figures/h2_training_fraction.pdf",
    bbox_inches="tight",
)

plt.savefig(
    "figures/h2_training_fraction.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FIGURE 3 — H4 OVERALL SEVERITY TREND
# ============================================================

files = [
    "results/blood_corruption_results.csv",
    "results/derma_corruption_results.csv",
    "results/pneumonia_corruption_results.csv",
]

corr = pd.concat(
    [pd.read_csv(f) for f in files],
    ignore_index=True,
)


severity = (
    corr.groupby(
        [
            "dataset",
            "condition",
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


sev_gap = pre.merge(
    scr,
    on=[
        "dataset",
        "severity",
    ],
    suffixes=("_pre", "_scratch"),
)


sev_gap["gap"] = (
    sev_gap["corrupted_bacc_pre"]
    - sev_gap["corrupted_bacc_scratch"]
) * 100


plt.figure(figsize=(8, 5))

for dataset in datasets:

    temp = (
        sev_gap[
            sev_gap["dataset"] == dataset
        ]
        .sort_values("severity")
    )

    plt.plot(
        temp["severity"],
        temp["gap"],
        marker="o",
        label=dataset.capitalize(),
    )


plt.axhline(
    0,
    linewidth=1,
)

plt.xlabel(
    "Corruption severity"
)

plt.ylabel(
    "Pretrained − Scratch bACC (percentage points)"
)

plt.title(
    "Pretraining advantage across corruption severity"
)

plt.xticks(
    [1, 2, 3, 4, 5]
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "figures/h4_severity_trend.pdf",
    bbox_inches="tight",
)

plt.savefig(
    "figures/h4_severity_trend.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FIGURE 4 — H3 CLEAN GAIN VS CORRUPTED GAIN
# ============================================================

h3 = pd.read_csv(
    "results/h3_paired_effects.csv"
)


plt.figure(figsize=(7, 5))

for dataset in datasets:

    temp = h3[
        h3["dataset"] == dataset
    ]

    plt.scatter(
        temp["clean_gain"] * 100,
        temp["corrupted_gain"] * 100,
        label=dataset.capitalize(),
    )


plt.axhline(
    0,
    linewidth=1,
)

plt.axvline(
    0,
    linewidth=1,
)

plt.xlabel(
    "Clean accuracy gain (percentage points)"
)

plt.ylabel(
    "Corrupted accuracy gain (percentage points)"
)

plt.title(
    "Clean accuracy gain does not predict corruption robustness"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "figures/h3_clean_vs_corrupted.pdf",
    bbox_inches="tight",
)

plt.savefig(
    "figures/h3_clean_vs_corrupted.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


print("Figures created successfully.")

print("\nFiles:")

for f in sorted(
    os.listdir("figures")
):
    print(f)
