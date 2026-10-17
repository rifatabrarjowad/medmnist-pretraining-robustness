import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(
    "results/h3_paired_effects.csv"
)

plt.figure(figsize=(8, 5))

for dataset in [
    "blood",
    "derma",
    "pneumonia",
]:
    temp = df[
        df["dataset"] == dataset
    ]

    plt.scatter(
        temp["clean_gain"] * 100,
        temp["corrupted_gain"] * 100,
        label=dataset.capitalize(),
        s=55,
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
    "Clean accuracy gain versus corrupted accuracy gain"
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

print("Regenerated Figure 3.")
