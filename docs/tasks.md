# Exposomic Trajectories of Cognitive Decline in NHANES — Improvement Tasks

Below is an ordered, actionable checklist to improve the architecture, reliability, performance, and usability of the project. Each item is a concrete task that can be checked off upon completion.

1. [x] Establish a clear module responsibility map and high-level architecture diagram in docs/architecture.md.
2. [x] Introduce a configuration system (pydantic-settings or dynaconf) for paths, constants, and toggles currently hard-coded in CLI and modules.
3. [x] Refactor CLI (src/excog_trajectory/cli.py) into subcommands organized by feature with per-command functions and argument groups to reduce file size and cognitive load.
4. [x] Add logging (Python logging) with structured, leveled logs across modules; replace print calls in CLI and data processing with logger usage.
5. [x] Centralize constants currently in columns.py and scattered literals into a dedicated config/constants module, with docstrings and validation.
6. [x] Add input validation and error handling (custom exceptions) for user-facing functions in cli.py, data.py, analysis.py, trajectory.py, and visualization.py.
7. [x] Define public APIs via __all__ and docstrings for each module to clarify what’s supported.
8. [x] Add type hints across all functions and enable mypy strict checking; fix all typing issues.
9. [x] Ensure functions use precise types (e.g., pd.DataFrame, pd.Series, np.ndarray) and avoid Union where not necessary by splitting responsibilities.
10. [x] Introduce dataclasses or TypedDicts for structured return types (e.g., PLSR results, SNF outputs) instead of loosely structured dicts or tuples.
11. [x] Reduce function length and complexity (e.g., cli.clean_data, data.apply_qc_rules, data.impute_exposure_variables) by extracting helpers.
12. [x] Move plotting layout and style configuration to a small theming utility (visualization theme) to DRY repeated parameters and styles.
13. [x] Validate column dependencies early with explicit, descriptive errors (wrapping columns.validate_columns where used), including suggestions for remediation.
14. [x] Add a data schema contract (pandera or pydantic models) for input/output DataFrames at key pipeline steps (post-cleaning, pre-imputation, post-imputation).
15. [x] Document quality control (QC) rules comprehensively in README and a new docs/qc_rules.md, including references and rationales.
16. [x] Implement deterministic pipelines by threading random_state across all stochastic components (miceforest, CV splits, permutations) and documenting seeds.
17. [x] Add caching for expensive steps (e.g., imputation kernels, PLSR CV outcomes) with versioned cache keys and invalidation strategy.
18. [x] Parallelization review: ensure multiprocessing usage in analysis.pls_double_cv is safe on all platforms; guard entry-points with if __name__ == "__main__" for scripts.
19. [x] Optimize RRPP (trajectory.RRPP) by vectorizing loops and allowing configurable permutations with progress and early stopping for stability checks.
20. [x] Confirm StandardScaler usage is appropriate and consistent; centralize scaling pipelines to avoid leakage and duplicated logic.
21. [ ] Add unit tests for columns helpers (validate_columns, get_dummy_vars, get_exposure_vars) covering edge cases (missing columns, overlapping prefixes).
22. [ ] Add unit tests for data QC functions (apply_qc_rules, categorize_dsst_by_age) using synthetic inputs and boundary conditions.
23. [ ] Add unit tests for analysis.pls_double_cv, including a small synthetic dataset to validate component selection and R2 computation.
24. [ ] Add unit tests for trajectory functions (_get_ls_vectors, estimate_betas, estimate_difference, transform_vectors_to_original), including shape checks and invariants.
25. [ ] Add unit tests for visualization functions to ensure files are created and basic figure properties (use non-interactive backend and temporary directories).
26. [ ] Create integration tests for CLI workflows: clean, download (mock network), impute (short iterations), plsr, snf, and trajectory.
27. [ ] Mock external dependencies (network calls in data.download_nhanes_data, file I/O) in tests; avoid writing to project root during tests.
28. [ ] Add test fixtures for small example datasets and expected outputs in tests/fixtures/.
29. [ ] Adopt Black formatting and isort consistently; add pre-commit hooks for black, isort, ruff, and mypy.
30. [ ] Configure ruff with E, F, B, I rule sets as per guidelines; fix all lint warnings and errors.
31. [ ] Ensure mypy passes with strict settings; add necessary type stubs or protocol definitions for libraries as needed.
32. [ ] Add docstrings (numpy or Google style) and usage examples to all public functions.
33. [ ] Expand README with quickstart examples for each CLI command and a minimal end-to-end workflow.
34. [ ] Create CLI help examples and --example-config output to generate a sample config file for users.
35. [ ] Add a top-level design doc for the statistical approaches (PLSR, SNF, RRPP, trajectory comparison) with references and assumptions in docs/methods.md.
36. [ ] Create a troubleshooting guide (docs/troubleshooting.md) for common errors (missing columns, file not found, imputation kernel issues).
37. [ ] Ensure all file paths resolve relative to project root or configurable data directory; avoid accidental writes to CWD.
38. [ ] Replace os.path with pathlib.Path consistently for path handling and improved readability.
39. [ ] Improve error messages for file not found in CLI commands, suggesting how to generate or where to download data.
40. [ ] Add progress bars (tqdm) for long-running tasks (imputation, permutations, cross-validation), with a quiet mode for CI.
41. [ ] Add reproducible environment files: environment.yml or uv lock/update instructions in README, and pin dev dependencies.
42. [ ] Add CI (GitHub Actions) to run: ruff, black --check, isort --check, mypy, and pytest with coverage; badge in README.
43. [ ] Collect code coverage with pytest-cov and set a minimum threshold (e.g., 75%) enforced by CI.
44. [ ] Implement graceful interruption handling (KeyboardInterrupt) for long computations, persisting partial results when sensible.
45. [ ] Validate and sanitize user-provided inputs to CLI (e.g., integers > 0, file existence, category lists non-empty) with argparse type/choices.
46. [ ] Ensure trajectory.create_model_matrix and related functions robustly handle mismatched dummy variables across datasets and provide guidance to users.
47. [ ] Add consistent random sub-sampling or stratification options to CV splits (e.g., by cycles or sex where appropriate), documented.
48. [ ] Review permutation p-value computation (p = (r+1)/(n+1)) to avoid zero p-values and add confidence intervals/bias correction notes.
49. [ ] Expose programmatic API equivalents for CLI commands (thin wrappers) to improve reuse in notebooks and scripts.
50. [ ] Provide example notebooks in examples/ demonstrating end-to-end usage on a small sample dataset with synthetic or public data.
51. [ ] Introduce a lightweight dependency injection approach for data sources to facilitate testing and swapping real vs. mock data.
52. [ ] Add input/output versioning metadata to processed files (e.g., in CSV headers or sidecar JSON) to detect stale artifacts.
53. [ ] Validate that visualization functions do not assume presence of specific columns; add guardrails and friendly messages when data is insufficient.
54. [ ] Add safe file naming utilities to sanitize output file names and avoid collisions; include timestamp/UUID options.
55. [ ] Review memory usage for large DataFrames (e.g., imputation); add chunking or on-disk strategies where feasible.
56. [ ] Ensure SNF implementation (analysis.run_snf) is correct: verify kNN, normalization, diffusion steps; add citations and tests against known behavior.
57. [ ] Add feature importance interpretation guidance for VIP scores: thresholds, caveats, and stability checks across resamples.
58. [ ] Validate standardization scope (fit/train vs. apply/test) in CV to prevent leakage; restructure code to use scikit-learn Pipelines where appropriate.
59. [ ] Add consistent random seeds and document in outputs metadata for reproducibility (e.g., JSON sidecars accompanying results).
60. [ ] Publish a small synthetic dataset under data/processed/example/ for quick validation runs and demos; wire into tests.
61. [ ] Review and fix any duplication (e.g., duplicate nested defs in analysis.run_snf structure per file structure) and ensure function boundaries are clear.
62. [ ] Replace magic numbers (permutation counts, CV splits, top-N VIPs) with configurable parameters across API and CLI.
63. [ ] Add graceful handling when model artifacts (pickles/kernels) are missing or incompatible; include version checks and helpful remediation guidance.
64. [ ] Harden serialization: prefer joblib for large numpy arrays and scikit-learn models; document security caveats of loading pickles.
65. [ ] Add security review: avoid executing arbitrary code paths, validate URLs for downloads, and ensure safe temp file handling.
66. [ ] Introduce a simple plugin mechanism or registry for new exposure groups or cognitive measures to reduce hard-coding.
67. [ ] Improve doc build process: generate API docs (pdoc or Sphinx) and host locally; ensure docstrings render examples.
68. [ ] Provide performance benchmarks and a reproducible benchmarking script under benchmarks/ for key routines (imputation, RRPP, PLSR CV).
69. [ ] Add metadata to figures (titles, captions, data notes) and standardize export formats (PNG + SVG) with consistent DPI.
70. [ ] Add command to CLI for diagnostics (data summary, missingness report, variable types) and export as HTML/Markdown.
71. [ ] Ensure columns IDS/COGNITIVE_VARS/COVARIATES/CATEGORICAL_COVARIATES are documented and configurable per NHANES cycles.
72. [ ] Implement careful NaN handling checks throughout; add explicit assertions before fitting models.
73. [ ] Add graceful degradation paths when optional dependencies (e.g., miceforest) are not installed; provide informative install hints.
74. [ ] Provide explicit licenses and citation information in README; add CITATION.cff.
75. [ ] Validate packaging metadata in pyproject.toml (dependencies, entry points, classifiers) and add console_scripts for CLI.
76. [ ] Add make commands or justfile tasks for common actions (format, lint, type-check, test, build, docs).
77. [ ] Ensure tests and scripts respect platform differences (Windows paths, multiprocessing start method, encoding).
78. [ ] Add stateful run-IDs and structured result directories in results/ to avoid overwrites and aid provenance.
79. [ ] Create a deprecation policy and guidelines for changes to public APIs, documenting in CONTRIBUTING.md.
80. [ ] Add a small health check CI job that runs a smoke test CLI pipeline on the sample dataset within time/memory constraints.
