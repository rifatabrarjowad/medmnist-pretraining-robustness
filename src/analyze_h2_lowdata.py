import numpy as np
import pandas as pd


N_BOOT = 10_000
BOOT_SEED = 2026


def bootstrap_mean_ci(values, seed):
    values = np.asarray(values, dtype=float)

    rng = np.random.default_rng(seed)

    samples = rng.choice(
        values,
        size=(N_BOOT, len(values)),
        replace=True,
    )

    means = samples.mean(axis=1)

    return np.percentile(
        means,
        [2.5, 97.5],
    )


# ============================================================
# CLEAN PERFORMANCE
# ============================================================

clean = pd.read_csv(
    "results/blood_ablation_clean_test.csv"
)

clean_wide = clean.pivot(
    index=["fraction", "seed"],
    columns="condition",
    values="test_bacc",
).reset_index()

clean_wide["gap"] = (
    clean_wide["pretrained"]
    - clean_wide["scratch"]
)

clean_100 = (
    clean_wide[
        clean_wide["fraction"] == 1.0
    ]
    .set_index("seed")["gap"]
)

clean_10 = (
    clean_wide[
        clean_wide["fraction"] == 0.1
    ]
    .set_index("seed")["gap"]
)

common_seeds = clean_100.index.intersection(
    clean_10.index
)

clean_change = (
    clean_10.loc[common_seeds]
    - clean_100.loc[common_seeds]
)

clean_ci = bootstrap_mean_ci(
    clean_change.to_numpy(),
    BOOT_SEED,
)


# ============================================================
# CORRUPTED PERFORMANCE
# ============================================================

corr = pd.read_csv(
    "results/blood_ablation_corruption_results.csv"
)

# Average over all corruption × severity values
# within each fraction × condition × seed.
corr_seed = (
    corr.groupby(
        ["fraction", "condition", "seed"],
        as_index=False,
    )["corrupted_bacc"]
    .mean()
)

corr_wide = corr_seed.pivot(
    index=["fraction", "seed"],
    columns="condition",
    values="corrupted_bacc",
).reset_index()

corr_wide["gap"] = (
    corr_wide["pretrained"]
    - corr_wide["scratch"]
)

corr_100 = (
    corr_wide[
        corr_wide["fraction"] == 1.0
    ]
    .set_index("seed")["gap"]
)

corr_10 = (
    corr_wide[
        corr_wide["fraction"] == 0.1
    ]
    .set_index("seed")["gap"]
)

common_seeds_corr = corr_100.index.intersection(
    corr_10.index
)

corr_change = (
    corr_10.loc[common_seeds_corr]
    - corr_100.loc[common_seeds_corr]
)

corr_ci = bootstrap_mean_ci(
    corr_change.to_numpy(),
    BOOT_SEED + 1,
)


# ============================================================
# OUTPUT
# ============================================================

print("\n========================================")
print("H2 LOW-DATA PAIRED ANALYSIS")
print("10% gap minus 100% gap")
print("positive = pretraining advantage increased")
print("========================================")

print("\nCLEAN")
print("Seed-wise changes:")
for seed, value in clean_change.items():
    print(
        f"  seed {seed}: "
        f"{value * 100:+.2f} pp"
    )

print(
    f"Mean change: "
    f"{clean_change.mean() * 100:+.2f} pp"
)

print(
    f"Bootstrap 95% CI: "
    f"[{clean_ci[0] * 100:+.2f}, "
    f"{clean_ci[1] * 100:+.2f}]"
)


print("\nCORRUPTED")
print("Seed-wise changes:")
for seed, value in corr_change.items():
    print(
        f"  seed {seed}: "
        f"{value * 100:+.2f} pp"
    )

print(
    f"Mean change: "
    f"{corr_change.mean() * 100:+.2f} pp"
)

print(
    f"Bootstrap 95% CI: "
    f"[{corr_ci[0] * 100:+.2f}, "
    f"{corr_ci[1] * 100:+.2f}]"
)


summary = pd.DataFrame([
    {
        "endpoint": "clean",
        "n_seeds": len(clean_change),
        "mean_change_10_minus_100":
            clean_change.mean(),
        "ci_low": clean_ci[0],
        "ci_high": clean_ci[1],
    },
    {
        "endpoint": "corrupted",
        "n_seeds": len(corr_change),
        "mean_change_10_minus_100":
            corr_change.mean(),
        "ci_low": corr_ci[0],
        "ci_high": corr_ci[1],
    },
])

summary.to_csv(
    "results/h2_lowdata_paired.csv",
    index=False,
)

print("\nSaved:")
print("results/h2_lowdata_paired.csv")
