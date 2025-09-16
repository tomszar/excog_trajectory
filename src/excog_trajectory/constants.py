"""
Centralized constants for default paths, filenames, and URLs.

This module consolidates default values used across the CLI and modules to
reduce duplication and improve discoverability.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final, List

# Base directories (relative to project root by default)
DATA_DIR: Final[Path] = Path("data")
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
RESULTS_DIR: Final[Path] = Path("results")

# Filenames
DEFAULT_NHANES_FILENAME: Final[str] = "nhanes_data.csv"

# CLI defaults
DEFAULT_DOWNLOAD_OUTPUT_DIR: Final[str] = str(RAW_DATA_DIR)
DEFAULT_CLEAN_OUTPUT_DIR: Final[str] = str(RESULTS_DIR)
DEFAULT_CLEAN_OUTPUT_DATA: Final[str] = str(PROCESSED_DATA_DIR)
DEFAULT_IMPUTE_DATA_PATH: Final[str] = str(PROCESSED_DATA_DIR / "cleaned_nhanes.csv")
DEFAULT_IMPUTE_OUTPUT_PATH: Final[str] | None = None
DEFAULT_PLSR_DATA_PATH: Final[str] = str(PROCESSED_DATA_DIR / "imputed") + "/"
DEFAULT_PLSR_OUTPUT_DIR: Final[str] = str(RESULTS_DIR / "plsr")
DEFAULT_SNF_DATA_PATH: Final[str] = str(PROCESSED_DATA_DIR / "imputed_nhanes_dat1.csv")
DEFAULT_SNF_OUTPUT_DIR: Final[str] = str(RESULTS_DIR / "snf")
DEFAULT_TRAJECTORY_DATA_PATH: Final[str] = str(PROCESSED_DATA_DIR / "imputed" / "imputed_nhanes_dat1.csv")
DEFAULT_TRAJECTORY_MODEL_PATH: Final[str] = str(RESULTS_DIR / "plsr" / "best_model.pkl")
DEFAULT_TRAJECTORY_OUTPUT_DIR: Final[str] = str(RESULTS_DIR / "trajectory")

# URLs
DEFAULT_DIRECT_URLS: Final[List[str]] = [
    "https://osf.io/download/9aupq/",
    "https://osf.io/download/9vewm/",
]

__all__ = [
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "RESULTS_DIR",
    "DEFAULT_NHANES_FILENAME",
    "DEFAULT_DOWNLOAD_OUTPUT_DIR",
    "DEFAULT_CLEAN_OUTPUT_DIR",
    "DEFAULT_CLEAN_OUTPUT_DATA",
    "DEFAULT_IMPUTE_DATA_PATH",
    "DEFAULT_IMPUTE_OUTPUT_PATH",
    "DEFAULT_PLSR_DATA_PATH",
    "DEFAULT_PLSR_OUTPUT_DIR",
    "DEFAULT_SNF_DATA_PATH",
    "DEFAULT_SNF_OUTPUT_DIR",
    "DEFAULT_TRAJECTORY_DATA_PATH",
    "DEFAULT_TRAJECTORY_MODEL_PATH",
    "DEFAULT_TRAJECTORY_OUTPUT_DIR",
    "DEFAULT_DIRECT_URLS",
]