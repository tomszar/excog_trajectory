"""
Module for managing column names in the excog_trajectory package.

This module provides constants and utilities for handling column names,
validating their existence in datasets, and managing dummy variables.
"""

from typing import List, Optional

import pandas as pd

# Define constants for column categories
IDS = ["SEQN", "sample"]
COGNITIVE_VARS = ["CFDRIGHT", "CFDDS"]
COGNITIVE_CAT = ["DSST_High", "DSST_Average", "DSST_Low",]
COVARIATES = ["Cycle", "RIDAGEYR", "RIAGENDR", "INDFMPIR", "DMDEDUC2", "RIDRETH1"]
CATEGORICAL_COVARIATES = ["Cycle", "RIAGENDR", "RIDRETH1"]
COLS_TO_DROP = ["SDDSRVYR", "INDHHINC", "INDHHIN2"]


def validate_columns(
        data: pd.DataFrame,
        columns: List[str],
        raise_error: bool = True) -> List[str]:
    """
    Validate that cols exist in the DataFrame.

    Args:
        data: DataFrame to check
        columns: List of column names to validate
        raise_error: Whether to raise an error if cols are missing

    Returns:
        List of valid cols that exist in the DataFrame

    Raises:
        ValueError: If raise_error is True and any cols are missing
    """
    missing_columns = [col for col in columns if col not in data.columns]
    valid_columns = [col for col in columns if col in data.columns]

    if missing_columns and raise_error:
        raise ValueError(f"The following cols are missing from the DataFrame: {missing_columns}")

    return valid_columns


def get_dummy_prefixes(categorical_columns: Optional[List[str]] = None) -> List[str]:
    """
    Get prefixes for dummy variables based on categorical cols.

    Args:
        categorical_columns: List of categorical column names

    Returns:
        List of prefixes for dummy variables
    """
    if categorical_columns is None:
        categorical_columns = CATEGORICAL_COVARIATES

    return [f"{col}_" for col in categorical_columns]


def get_dummy_vars(data: pd.DataFrame,
                   categorical_covariates: Optional[List[str]] = None) -> List[str]:
    """
    Get the names of all dummy variables in the DataFrame.

    Args:
        data: DataFrame containing the data
        categorical_covariates: List of categorical covariate names

    Returns:
        List of dummy variable names
    """
    if categorical_covariates is None:
        categorical_covariates = CATEGORICAL_COVARIATES

    # Get dummy prefixes
    dummy_prefixes = get_dummy_prefixes(categorical_covariates)

    # Find all columns that start with any of the dummy prefixes
    dummy_vars = [col for col in data.columns
                  if any(col.startswith(prefix) for prefix in dummy_prefixes)]

    return dummy_vars


def get_exposure_vars(data: pd.DataFrame,
                      cognitive_vars: Optional[List[str]] = None,
                      covariates: Optional[List[str]] = None,
                      categorical_covariates: Optional[List[str]] = None,
                      id_vars: Optional[List[str]] = None) -> List[str]:
    """
    Get exposure variables by excluding cognitive variables, covariates, and dummy variables.

    Args:
        data: DataFrame containing the data
        cognitive_vars: List of cognitive variable names
        covariates: List of covariate names
        categorical_covariates: List of categorical covariate names

    Returns:
        List of exposure variable names
    """
    if cognitive_vars is None:
        cognitive_vars = COGNITIVE_VARS + COGNITIVE_CAT

    if covariates is None:
        covariates = COVARIATES

    if categorical_covariates is None:
        categorical_covariates = CATEGORICAL_COVARIATES

    if id_vars is None:
        id_vars = IDS

    # Validate cols
    valid_cognitive_vars = validate_columns(data, cognitive_vars, raise_error=False)
    valid_covariates = validate_columns(data, covariates, raise_error=False)
    valid_id_vars = validate_columns(data, id_vars, raise_error=False)

    # Get dummy prefixes
    dummy_prefixes = get_dummy_prefixes(categorical_covariates)

    # Filter out covariates and any column that starts with dummy variable prefixes
    exposure_vars = [col for col in data.columns
                     if col not in valid_cognitive_vars + valid_covariates + valid_id_vars and
                     not any(col.startswith(prefix) for prefix in dummy_prefixes)]

    return exposure_vars
