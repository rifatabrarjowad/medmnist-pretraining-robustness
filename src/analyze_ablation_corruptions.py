import numpy as np
import pandas as pd


N_BOOT = 10_000
BOOT_SEED = 2026

df = pd.read_csv(
    "results/blood_ablation_corruption_results.csv"
)

# First average across corruptions and severities
# within each fraction × condition × seed.
seed_summary = (
    df.groupby(
        [
            "fraction",
            "condition",
            "seed",
        ],
        as_index=False,
    )["corrupted_bacc"]
    .mean()
)

# Put pretrained and scratch on same row.
paired = seed_summary.pivot(
    index=[
        "fraction",
        "seed",
    ],
    columns="condition",
    values="corrupted_bacc",
).reset_index()

paired["gap"] = (
    paired["pretrained"]
    - paired["scratch"]
)


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


rows = []

for fraction, group in paired.groupby("fraction"):

    gaps = group["gap"].to_numpy()

    ci_low, ci_high = bootstrap_mean_ci(
        gaps,
        seed=BOOT_SEED + int(fraction * 100),
    )

    rows.append({
        "fraction": fraction,
        "n_seeds": len(gaps),
        "pretrained_mean_corrupted_bacc":
            group["pretrained"].mean(),
        "scratch_mean_corrupted_bacc":
            group["scratch"].mean(),
        "mean_gap":
            gaps.mean(),
        "sd_gap":
            gaps.std(ddof=1),
        "ci_low":
            ci_low,
        "ci_high":
            ci_high,
    })


out = pd.DataFrame(rows).sort_values(
    "fraction",
    ascending=False,
)

out.to_csv(
    "results/blood_ablation_corruption_gap_ci.csv",
    index=False,
)

print("\n====================================")
print("BLOOD ABLATION — CORRUPTION RESULTS")
print("positive gap = pretrained better")
print("====================================")

for _, row in out.iterrows():

    print(
        f"{row['fraction'] * 100:>5.0f}% "
        f"| pretrained={row['pretrained_mean_corrupted_bacc']:.4f} "
        f"| scratch={row['scratch_mean_corrupted_bacc']:.4f} "
        f"| gap={row['mean_gap'] * 100:+.2f} pp "
        f"| 95% CI "
        f"[{row['ci_low'] * 100:+.2f}, "
        f"{row['ci_high'] * 100:+.2f}]"
    )

print("\nSaved:")
print("results/blood_ablation_corruption_gap_ci.csv")
