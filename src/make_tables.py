import pandas as pd


# ============================================================
# TABLE 1 — CLEAN + CORRUPTED PERFORMANCE
# ============================================================

h3 = pd.read_csv(
    "results/h3_condition_summary.csv"
)

table1 = h3[
    [
        "dataset",
        "condition",
        "clean_mean",
        "clean_sd",
        "corrupted_mean",
        "corrupted_sd",
        "BE_mean",
        "BE_sd",
        "rBE_mean",
        "rBE_sd",
    ]
].copy()


for col in [
    "clean_mean",
    "clean_sd",
    "corrupted_mean",
    "corrupted_sd",
]:
    table1[col] = table1[col] * 100


table1.columns = [
    "Dataset",
    "Initialization",
    "Clean bACC mean (%)",
    "Clean bACC SD",
    "Corrupted bACC mean (%)",
    "Corrupted bACC SD",
    "BE mean",
    "BE SD",
    "rBE mean",
    "rBE SD",
]


table1.to_csv(
    "results/table1_main_performance.csv",
    index=False,
)


print("\n======================================")
print("TABLE 1 — MAIN PERFORMANCE")
print("======================================")

print(
    table1.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


# ============================================================
# TABLE 2 — HYPOTHESIS SUMMARY
# ============================================================

table2 = pd.DataFrame(
    [
        {
            "Hypothesis": "H1",
            "Prediction":
                "Task-specific pretraining benefit smaller than noise/blur/digital",
            "Key result":
                "Interaction reversed: Blood +1.0009; Derma +0.5657",
            "Verdict": "Rejected",
        },
        {
            "Hypothesis": "H2",
            "Prediction":
                "Pretraining benefit increases as training data decreases",
            "Key result":
                "Gap increased from +0.47 pp at 100% to +3.72 pp at 10%",
            "Verdict": "Supported",
        },
        {
            "Hypothesis": "H3",
            "Prediction":
                "Clean accuracy gain explains much of robustness gain",
            "Key result":
                "Clean gain negatively associated with corrupted gain",
            "Verdict": "Rejected",
        },
        {
            "Hypothesis": "H4",
            "Prediction":
                "Pretraining benefit decreases monotonically with severity",
            "Key result":
                "Only 2/13 family cells were monotonically decreasing",
            "Verdict": "Rejected",
        },
    ]
)


table2.to_csv(
    "results/table2_hypothesis_summary.csv",
    index=False,
)


print("\n======================================")
print("TABLE 2 — HYPOTHESIS SUMMARY")
print("======================================")

print(
    table2.to_string(
        index=False
    )
)


# ============================================================
# TABLE 3 — FAMILY EFFECTS
# ============================================================

family = pd.read_csv(
    "results/family_summary.csv"
)

table3 = family.copy()

table3["mean_delta_bacc_pp"] = (
    table3["mean_delta_bacc"] * 100
)

table3 = table3[
    [
        "dataset",
        "family",
        "n_corruptions",
        "mean_delta_bacc_pp",
        "mean_delta_BE",
    ]
]

table3.columns = [
    "Dataset",
    "Corruption family",
    "N corruptions",
    "Δ bACC (pp)",
    "Δ BE",
]


table3.to_csv(
    "results/table3_family_effects.csv",
    index=False,
)


print("\n======================================")
print("TABLE 3 — FAMILY EFFECTS")
print("positive = pretrained better")
print("======================================")

print(
    table3.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


print("\nSaved:")
print("results/table1_main_performance.csv")
print("results/table2_hypothesis_summary.csv")
print("results/table3_family_effects.csv")
