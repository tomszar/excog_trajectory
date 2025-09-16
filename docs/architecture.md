# Architecture and Module Responsibilities

This document provides a high-level view of the excog-trajectory project and clarifies responsibilities for each module.

## High-Level Overview

The project analyzes exposomic trajectories of cognitive decline using NHANES data. It provides a command-line interface (CLI) for end-to-end workflows and a programmatic API for individual steps.

```
+--------------------+       +----------------+       +-------------------+
|  Data Acquisition  |  -->  |  Data Cleaning |  -->  |  Imputation (MICE)|
| (download/extract) |       |  & QC Rules    |       |  miceforest       |
+--------------------+       +----------------+       +-------------------+
           |                            |                        |
           v                            v                        v
     +-----------+               +-------------+         +--------------+
     |  Analysis | <-----------> | Trajectory  |  and    | Visualization |
     | (PLSR/SNF)|               |  Comparison |  --->   |  (figures)    |
     +-----------+               +-------------+         +--------------+
             ^                           ^                         |
             |                           |                         v
             +---------------------------+-------------------------+
                                     CLI
```

- CLI orchestrates workflows across modules.
- Data module handles loading, cleaning/QC, imputation, and basic transformations.
- Analysis module implements modeling routines (PLSR, SNF) and related utilities.
- Trajectory module compares cognitive decline trajectories (RRPP, vector transforms).
- Visualization module creates diagnostic and publication-quality plots.
- Columns module centralizes column naming logic and helpers.
- New supporting modules (config, constants, logging_utils, exceptions, types) improve maintainability and usability.

## Module Responsibilities

- src/excog_trajectory/cli.py
  - Exposes subcommands: download, clean, impute, plsr, snf, trajectory
  - Parses arguments, validates inputs, calls into data/analysis/trajectory/visualization
  - Emits structured logs and user-friendly messages

- src/excog_trajectory/data.py
  - Download/extract NHANES data
  - Apply QC rules, categorization, filtering
  - Imputation with miceforest; saving artifacts and derived datasets

- src/excog_trajectory/analysis.py
  - PLSR double cross-validation and VIP computation
  - Similarity Network Fusion (SNF) utilities

- src/excog_trajectory/trajectory.py
  - LS means, RRPP permutation testing
  - Vector transformations and comparisons across groups

- src/excog_trajectory/visualization.py
  - Correlation matrices, VIP plots, score plots, trajectory visuals

- src/excog_trajectory/columns.py
  - Column constants and helpers: validation, dummy variable detection, exposure selection

- src/excog_trajectory/config.py (new)
  - Application configuration (paths, toggles) with environment overrides

- src/excog_trajectory/constants.py (new)
  - Shared constants for defaults, filenames, directory names

- src/excog_trajectory/logging_utils.py (new)
  - Package-wide logger configuration and helpers

- src/excog_trajectory/exceptions.py (new)
  - Domain-specific exception hierarchy

- src/excog_trajectory/types.py (new)
  - Typed containers (dataclasses/TypedDicts) for structured results

## Data/Results Layout (defaults)

- data/raw: downloads and extracted sources
- data/processed: cleaned and imputed datasets
- results/*: analysis outputs (plsr, snf, trajectory, correlation_matrices)

## Notes

- Python >= 3.11 runtime; tooling targets py38 per pyproject (may be updated later).
- Random seeds are exposed via function args/CLI options.
- Long-running operations will progressively incorporate progress bars and caching (future tasks).
