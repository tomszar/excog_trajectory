"""
Unit tests for the analysis module.
"""

import pytest
import pandas as pd
import numpy as np
from excog_trajectory import analysis


def test_run_linear_models():
    """Test that run_linear_models returns a dictionary."""
    # Create a mock DataFrame
    test_data = pd.DataFrame({
        "outcome1": [1, 2, 3, 4, 5],
        "exposure1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "exposure2": [1.0, 2.0, 3.0, 4.0, 5.0],
        "covariate1": [10, 20, 30, 40, 50],
        "covariate2": [100, 200, 300, 400, 500]
    })

    # Call the function
    result = analysis.run_linear_models(
        data=test_data,
        outcome_vars=["outcome1"],
        exposure_vars=["exposure1", "exposure2"],
        covariates=["covariate1", "covariate2"]
    )

    # Check that the result is a dictionary
    assert isinstance(result, dict)


def test_run_longitudinal_analysis():
    """Test that run_longitudinal_analysis returns a dictionary."""
    # Create a mock DataFrame with longitudinal data
    test_data = pd.DataFrame({
        "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "time": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "outcome": [10, 11, 12, 20, 21, 22, 30, 31, 32],
        "exposure1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "covariate1": [100, 100, 100, 200, 200, 200, 300, 300, 300]
    })

    # Call the function
    result = analysis.run_longitudinal_analysis(
        data=test_data,
        outcome_var="outcome",
        exposure_vars=["exposure1"],
        time_var="time",
        id_var="id",
        covariates=["covariate1"]
    )

    # Check that the result is a dictionary
    assert isinstance(result, dict)


def test_run_mixture_analysis():
    """Test that run_mixture_analysis returns a dictionary."""
    # Create a mock DataFrame
    test_data = pd.DataFrame({
        "outcome1": [1, 2, 3, 4, 5],
        "exposure1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "exposure2": [1.0, 2.0, 3.0, 4.0, 5.0],
        "covariate1": [10, 20, 30, 40, 50]
    })

    # Call the function
    result = analysis.run_mixture_analysis(
        data=test_data,
        outcome_vars=["outcome1"],
        exposure_vars=["exposure1", "exposure2"],
        covariates=["covariate1"],
        n_components=2
    )

    # Check that the result is a dictionary
    assert isinstance(result, dict)


def test_calculate_exposure_indices():
    """Test that calculate_exposure_indices returns a DataFrame with the original data."""
    # Create a mock DataFrame
    test_data = pd.DataFrame({
        "exposure1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "exposure2": [1.0, 2.0, 3.0, 4.0, 5.0],
        "other_col": [10, 20, 30, 40, 50]
    })

    # Call the function
    result = analysis.calculate_exposure_indices(
        data=test_data,
        exposure_vars=["exposure1", "exposure2"],
        method="sum"
    )

    # Check that the result is a DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # Check that the result contains the original data
    assert len(result) == len(test_data)
    assert "exposure1" in result.columns
    assert "exposure2" in result.columns
    assert "other_col" in result.columns
    
    # Check that the original DataFrame was not modified
    assert id(result) != id(test_data)