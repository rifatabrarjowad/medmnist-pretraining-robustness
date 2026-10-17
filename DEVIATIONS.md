# Deviations and Post-hoc Clarifications

## October 17, 2026 — Revision audit

The original preregistration was committed before experimental runs on August 30, 2026 (commit `5b1f1b4`).

During manuscript revision, the following issues were identified and handled transparently:

1. **PneumoniaMNIST gamma corruption family mapping**
   - The preregistration described PneumoniaMNIST as having no task-specific corruptions.
   - During revision, inspection of the MedMNIST-C corruption definitions showed that `gamma_corr_up` and `gamma_corr_down` should be treated as task-specific corruptions.
   - These corruptions were therefore reclassified as task-specific in descriptive family summaries and severity analyses.
   - PneumoniaMNIST was **not** added retrospectively to the preregistered confirmatory H1 contrast, which remains restricted to BloodMNIST and DermaMNIST.

2. **Learning-rate selection sensitivity analysis**
   - The preregistered protocol specified a fixed pretrained learning rate of `1e-4` and a validation-based scratch search over `{1e-4, 3e-4, 1e-3}`.
   - During revision, the original scratch-only search was recognized as asymmetric and as using a preprocessing/search setup that did not exactly match the final core pipeline.
   - A post-hoc symmetric validation-only sensitivity search was therefore conducted on BloodMNIST seed 0 for both initialization conditions using the final preprocessing and training protocol.
   - The validation-optimal rates matched those used in the core experiment: `1e-4` for pretrained and `3e-4` for scratch.
   - This analysis is reported as post hoc and does not redefine the original preregistered procedure.

3. **Post-hoc analyses added during revision**
   - BloodMNIST corruption evaluation across training fractions.
   - H3 within-dataset and dataset-centered correlation analyses.
   - Chance-normalized degradation / retained-performance sensitivity analysis.
   - These analyses are explicitly labeled post hoc or descriptive in the manuscript and do not replace the preregistered hypothesis decisions.

No completed core seed was excluded as an outlier, and no preregistered hypothesis was retroactively redefined.

4. **BloodMNIST reduced-data subset sampling**
   - The preregistration specified class-stratified subsampling for the E4 training-fraction ablation.
   - The implemented training code instead used a deterministic seeded random permutation of the full BloodMNIST training indices and selected the first `floor(N × fraction)` examples.
   - Therefore, the reduced-data subsets were **not explicitly class-stratified**.
   - For a given seed and fraction, the same subset was used for pretrained and scratch conditions because subset generation depended only on the seed and fraction, not on initialization condition.
   - Because the same seeded permutation was reused across fractions, the 10%, 25%, and 50% subsets were nested prefixes of that permutation for each seed.
   - This implementation difference was identified during revision and is reported as a deviation from the preregistered sampling procedure.
