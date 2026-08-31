import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# ============================================================
# LOAD CLEAN RESULTS
# ============================================================

clean_files = [
    "results/blood_clean_test.csv",
    "results/derma_clean_test.csv",
    "results/pneumonia_clean_test.csv",
]

clean = pd.concat(
    [pd.read_csv(f) for f in clean_files],
    ignore_index=True,
)


# ============================================================
# LOAD CORRUPTION RESULTS
# ============================================================

corruption_files = [
    "results/blood_corruption_results.csv",
    "results/derma_corruption_results.csv",
    "results/pneumonia_corruption_results.csv",
]

corr = pd.concat(
    [pd.read_csv(f) for f in corruption_files],
    ignore_index=True,
)


# ============================================================
# 1. CORRUPTED BACC PER DATASET × CONDITION × SEED
#
# corrupted_bacc exists once per corruption × severity,
# so averaging all rows gives mean corrupted performance.
# ============================================================

corrupted_seed = (
    corr.groupby(
        [
            "dataset",
            "condition",
            "seed",
        ]
    )
    .agg(
        mean_corrupted_bacc=(
            "corrupted_bacc",
            "mean",
        )
    )
    .reset_index()
)


# ============================================================
# 2. BE/rBE ARE CORRUPTION-LEVEL VALUES REPEATED ACROSS
#    SEVERITIES.
#
# Deduplicate first.
# ============================================================

per_corruption = (
    corr.groupby(
        [
            "dataset",
            "condition",
            "seed",
            "corruption",
        ]
    )
    .agg(
        BE=("BE", "first"),
        rBE=("rBE", "first"),
    )
    .reset_index()
)


normalized_seed = (
    per_corruption.groupby(
        [
            "dataset",
            "condition",
            "seed",
        ]
    )
    .agg(
        mean_BE=("BE", "mean"),
        mean_valid_rBE=("rBE", "mean"),
        valid_rBE_n=("rBE", "count"),
    )
    .reset_index()
)


# ============================================================
# 3. MERGE CLEAN + CORRUPTED
# ============================================================

seed_results = (
    clean.merge(
        corrupted_seed,
        on=[
            "dataset",
            "condition",
            "seed",
        ],
    )
    .merge(
        normalized_seed,
        on=[
            "dataset",
            "condition",
            "seed",
        ],
    )
)


# Absolute degradation caused by corruption
seed_results["absolute_drop"] = (
    seed_results["clean_test_bacc"]
    - seed_results["mean_corrupted_bacc"]
)


seed_results.to_csv(
    "results/h3_seed_metrics.csv",
    index=False,
)


# ============================================================
# 4. CONDITION SUMMARY
# ============================================================

summary = (
    seed_results.groupby(
        [
            "dataset",
            "condition",
        ]
    )
    .agg(
        clean_mean=("clean_test_bacc", "mean"),
        clean_sd=("clean_test_bacc", "std"),

        corrupted_mean=(
            "mean_corrupted_bacc",
            "mean",
        ),
        corrupted_sd=(
            "mean_corrupted_bacc",
            "std",
        ),

        drop_mean=("absolute_drop", "mean"),
        drop_sd=("absolute_drop", "std"),

        BE_mean=("mean_BE", "mean"),
        BE_sd=("mean_BE", "std"),

        rBE_mean=("mean_valid_rBE", "mean"),
        rBE_sd=("mean_valid_rBE", "std"),
    )
    .reset_index()
)


print("\n========================================")
print("H3 CONDITION SUMMARY")
print("========================================")

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


summary.to_csv(
    "results/h3_condition_summary.csv",
    index=False,
)


# ============================================================
# 5. PRETRAINED - SCRATCH PAIRED EFFECTS
# ============================================================

pre = seed_results[
    seed_results["condition"] == "pretrained"
].copy()

scr = seed_results[
    seed_results["condition"] == "scratch"
].copy()


paired = pre.merge(
    scr,
    on=[
        "dataset",
        "seed",
    ],
    suffixes=("_pre", "_scratch"),
)


# Positive = pretrained has higher clean accuracy
paired["clean_gain"] = (
    paired["clean_test_bacc_pre"]
    - paired["clean_test_bacc_scratch"]
)


# Positive = pretrained has higher corrupted accuracy
paired["corrupted_gain"] = (
    paired["mean_corrupted_bacc_pre"]
    - paired["mean_corrupted_bacc_scratch"]
)


# Positive = pretrained loses LESS accuracy under corruption
paired["drop_advantage"] = (
    paired["absolute_drop_scratch"]
    - paired["absolute_drop_pre"]
)


# Positive = pretrained has LOWER BE = better robustness
paired["BE_advantage"] = (
    paired["mean_BE_scratch"]
    - paired["mean_BE_pre"]
)


# Positive = pretrained has LOWER rBE
paired["rBE_advantage"] = (
    paired["mean_valid_rBE_scratch"]
    - paired["mean_valid_rBE_pre"]
)


paired.to_csv(
    "results/h3_paired_effects.csv",
    index=False,
)


print("\n========================================")
print("H3 PAIRED EFFECTS")
print("positive = pretrained better")
print("========================================")


for dataset in [
    "blood",
    "derma",
    "pneumonia",
]:

    temp = paired[
        paired["dataset"] == dataset
    ]

    print(f"\n--- {dataset.upper()} ---")

    for metric in [
        "clean_gain",
        "corrupted_gain",
        "drop_advantage",
        "BE_advantage",
        "rBE_advantage",
    ]:

        values = temp[metric]

        print(
            f"{metric:18s} "
            f"mean={values.mean():+.4f} "
            f"SD={values.std():.4f}"
        )


# ============================================================
# 6. HOW MUCH OF CORRUPTED GAP TRACKS CLEAN GAP?
#
# Across 3 datasets × 5 paired seeds = 15 observations.
# This is descriptive, not an independent-sample
# significance claim.
# ============================================================

x = paired["clean_gain"].to_numpy()
y = paired["corrupted_gain"].to_numpy()


pearson_r, pearson_p = pearsonr(x, y)
spearman_r, spearman_p = spearmanr(x, y)


print("\n========================================")
print("CLEAN GAIN vs CORRUPTED GAIN")
print("3 datasets × 5 paired seeds = 15 points")
print("========================================")

print(
    f"Pearson r  = {pearson_r:+.4f} "
    f"(p={pearson_p:.6f})"
)

print(
    f"Spearman ρ = {spearman_r:+.4f} "
    f"(p={spearman_p:.6f})"
)


# ============================================================
# 7. CLEAN GAIN vs NORMALIZED ROBUSTNESS ADVANTAGE
# ============================================================

y_be = paired["BE_advantage"].to_numpy()

pearson_be_r, pearson_be_p = pearsonr(
    x,
    y_be,
)

spearman_be_r, spearman_be_p = spearmanr(
    x,
    y_be,
)


print("\n========================================")
print("CLEAN GAIN vs BE ADVANTAGE")
print("positive BE advantage = pretrained better")
print("========================================")

print(
    f"Pearson r  = {pearson_be_r:+.4f} "
    f"(p={pearson_be_p:.6f})"
)

print(
    f"Spearman ρ = {spearman_be_r:+.4f} "
    f"(p={spearman_be_p:.6f})"
)


# ============================================================
# 8. DATASET-LEVEL DECOMPOSITION
# ============================================================

print("\n========================================")
print("H3 DECOMPOSITION")
print("========================================")

decomp_rows = []

for dataset in [
    "blood",
    "derma",
    "pneumonia",
]:

    temp = paired[
        paired["dataset"] == dataset
    ]

    clean_gain = temp["clean_gain"].mean()
    corrupted_gain = temp["corrupted_gain"].mean()
    drop_advantage = temp["drop_advantage"].mean()

    decomp_rows.append({
        "dataset": dataset,
        "clean_gain": clean_gain,
        "corrupted_gain": corrupted_gain,
        "drop_advantage": drop_advantage,
    })

    print(f"\n{dataset.upper()}")

    print(
        f"Clean gain:       "
        f"{clean_gain:+.4f}"
    )

    print(
        f"Corrupted gain:   "
        f"{corrupted_gain:+.4f}"
    )

    print(
        f"Drop advantage:   "
        f"{drop_advantage:+.4f}"
    )


pd.DataFrame(
    decomp_rows
).to_csv(
    "results/h3_decomposition.csv",
    index=False,
)


print("\nSaved:")
print("results/h3_seed_metrics.csv")
print("results/h3_condition_summary.csv")
print("results/h3_paired_effects.csv")
print("results/h3_decomposition.csv")
