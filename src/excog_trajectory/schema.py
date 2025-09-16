"""
Lightweight DataFrame schema validators used at key pipeline steps.

These validators perform minimal column-set checks to provide early, helpful
errors without adding heavy dependencies.
"""
from __future__ import annotations

from typing import Iterable, List

import pandas as pd

from . import columns


def _missing(df: pd.DataFrame, required: Iterable[str]) -> List[str]:
    return [c for c in required if c not in df.columns]


def validate_post_clean_schema(df: pd.DataFrame) -> None:
    """Validate that cleaned data contains expected columns.

    Requires at least one cognitive variable and core covariates to be present.
    """
    required_any = columns.COGNITIVE_VARS
    if all(c not in df.columns for c in required_any):
        raise ValueError(
            "Post-clean schema invalid: no cognitive variables found. Expected one of "
            f"{required_any}. Consider verifying your cleaning step."
        )
    core_covs = [c for c in ["RIDAGEYR", "RIAGENDR"] if c in columns.COVARIATES]
    miss = _missing(df, core_covs)
    if miss:
        raise ValueError(
            f"Post-clean schema invalid: missing core covariates {miss}."
        )


def validate_pre_impute_schema(df: pd.DataFrame) -> None:
    """Validate that data is ready for imputation.

    Ensures ID and cycle columns are present if expected.
    """
    # These are optional but recommended; warn-like error on total absence of IDs.
    if not any(c in df.columns for c in columns.IDS):
        raise ValueError(
            "Pre-impute schema invalid: no ID column found (expected one of "
            f"{columns.IDS})."
        )


def validate_post_impute_schema(df: pd.DataFrame) -> None:
    """Validate that imputed data has no all-NA exposure columns."""
    exposure_vars = columns.get_exposure_vars(df)
    empty = [c for c in exposure_vars if df[c].isna().all()]
    if empty:
        raise ValueError(
            "Post-impute schema invalid: some exposure columns remain entirely missing: "
            f"{empty[:5]}{'...' if len(empty) > 5 else ''}"
        )


__all__ = [
    "validate_post_clean_schema",
    "validate_pre_impute_schema",
    "validate_post_impute_schema",
]
