"""
Lightweight configuration system for excog-trajectory.

Settings are derived from environment variables with sensible defaults
falling back to project-level constants. This keeps runtime flexible
without adding heavy dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import List

from . import constants


@dataclass(frozen=True)
class Settings:
    # Directories
    data_dir: Path = Path(os.getenv("EXCOG_DATA_DIR", constants.DATA_DIR))
    raw_data_dir: Path = Path(os.getenv("EXCOG_RAW_DATA_DIR", constants.RAW_DATA_DIR))
    processed_data_dir: Path = Path(
        os.getenv("EXCOG_PROCESSED_DATA_DIR", constants.PROCESSED_DATA_DIR)
    )
    results_dir: Path = Path(os.getenv("EXCOG_RESULTS_DIR", constants.RESULTS_DIR))

    # Download
    direct_urls: List[str] = tuple(
        os.getenv("EXCOG_DIRECT_URLS", ",".join(constants.DEFAULT_DIRECT_URLS)).split(",")
    )  # type: ignore[assignment]
    nhanes_filename: str = os.getenv(
        "EXCOG_NHANES_FILENAME", constants.DEFAULT_NHANES_FILENAME
    )

    # CLI defaults (as strings for argparse compatibility where relevant)
    clean_output_dir: str = os.getenv(
        "EXCOG_CLEAN_OUTPUT_DIR", constants.DEFAULT_CLEAN_OUTPUT_DIR
    )
    clean_output_data: str = os.getenv(
        "EXCOG_CLEAN_OUTPUT_DATA", constants.DEFAULT_CLEAN_OUTPUT_DATA
    )
    impute_data_path: str = os.getenv(
        "EXCOG_IMPUTE_DATA_PATH", constants.DEFAULT_IMPUTE_DATA_PATH
    )
    impute_output_path: str | None = os.getenv(
        "EXCOG_IMPUTE_OUTPUT_PATH",
        constants.DEFAULT_IMPUTE_OUTPUT_PATH if constants.DEFAULT_IMPUTE_OUTPUT_PATH else None,
    )
    plsr_data_path: str = os.getenv(
        "EXCOG_PLSR_DATA_PATH", constants.DEFAULT_PLSR_DATA_PATH
    )
    plsr_output_dir: str = os.getenv(
        "EXCOG_PLSR_OUTPUT_DIR", constants.DEFAULT_PLSR_OUTPUT_DIR
    )
    snf_data_path: str = os.getenv(
        "EXCOG_SNF_DATA_PATH", constants.DEFAULT_SNF_DATA_PATH
    )
    snf_output_dir: str = os.getenv(
        "EXCOG_SNF_OUTPUT_DIR", constants.DEFAULT_SNF_OUTPUT_DIR
    )
    trajectory_data_path: str = os.getenv(
        "EXCOG_TRAJECTORY_DATA_PATH", constants.DEFAULT_TRAJECTORY_DATA_PATH
    )
    trajectory_model_path: str = os.getenv(
        "EXCOG_TRAJECTORY_MODEL_PATH", constants.DEFAULT_TRAJECTORY_MODEL_PATH
    )
    trajectory_output_dir: str = os.getenv(
        "EXCOG_TRAJECTORY_OUTPUT_DIR", constants.DEFAULT_TRAJECTORY_OUTPUT_DIR
    )

    log_level: str = os.getenv("EXCOG_LOG_LEVEL", "INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


__all__ = ["Settings", "get_settings"]
