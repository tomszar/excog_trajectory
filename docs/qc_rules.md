# Quality Control (QC) Rules

This document summarizes the QC rules applied during data cleaning and preprocessing.

Rules applied (see `data.apply_qc_rules` for implementation details):

1. Remove variables with fewer than 200 non-NaN values.
2. Remove categorical variables (heuristic: <10 unique values) if any category has fewer than 200 observations.
3. Remove variables with >=90% of non-NaN values equal to zero.
4. Remove variables with 100% missing data in at least one survey cycle (uses `SDDSRVYR` or `Cycle` dummies).

Additional processing:
- Categorical covariates are one-hot encoded (no drop-first) and retained as covariates.
- Optional log2 transform can be applied to exposure variables.
- Optional standardization (z-score) can be applied to exposures after QC.
- DSST cognitive score is categorized by age into DSST_High, DSST_Average, DSST_Low.

Notes and rationale:
- Thresholds are chosen for robustness in sparse exposure panels and mirror patterns from similar NHANES analyses.
- Rule 4 prevents leakage from cycles with entirely missing measurements.
- Log transform stabilizes right-skewed exposures; epsilon of 1e-10 avoids log(0).

Troubleshooting:
- If required columns are missing, see CLI error suggestions or run `excog clean` to regenerate cleaned datasets.
- Verify that your input files are aligned to expected NHANES cycles and variable names.
