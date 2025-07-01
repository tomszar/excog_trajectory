"""
Unit tests for the analysis module.
"""

import pytest
import pandas as pd
import numpy as np
from excog_trajectory import analysis
from sklearn.preprocessing import StandardScaler


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




def test_run_snf():
    """Test that run_snf returns a dictionary with the expected keys and shapes."""
    # Create a mock DataFrame
    test_data = pd.DataFrame({
        "exposure1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "exposure2": [1.0, 2.0, 3.0, 4.0, 5.0],
        "exposure3": [10, 20, 30, 40, 50],
        "exposure4": [100, 200, 300, 400, 500],
        "cognitive1": [1000, 2000, 3000, 4000, 5000],
        "covariate1": [10000, 20000, 30000, 40000, 50000],
        "covariate2": [100000, 200000, 300000, 400000, 500000]
    })

    # Define exposure categories
    exposure_categories = {
        "category1": ["exposure1", "exposure2"],
        "category2": ["exposure3", "exposure4"]
    }

    # Call the function
    result = analysis.run_snf(
        data=test_data,
        exposure_categories=exposure_categories,
        cognitive_vars=["cognitive1"],
        covariates=["covariate1", "covariate2"],
        k=2,  # Small k for test
        t=2,  # Small t for test
        alpha=0.5,
        scale=True
    )

    # Check that the result is a dictionary
    assert isinstance(result, dict)

    # Check that the result contains the expected keys
    expected_keys = [
        "fused_matrix", "similarity_matrices", "normalized_matrices",
        "k_nearest_matrices", "exposure_categories", "cognitive_vars",
        "covariates", "parameters"
    ]
    for key in expected_keys:
        assert key in result

    # Check that the shapes of the results are correct
    n_samples = len(test_data)
    assert result["fused_matrix"].shape == (n_samples, n_samples)

    # Check that the similarity matrices have the expected keys
    expected_similarity_keys = ["category1", "category2", "cognitive", "covariates"]
    for key in expected_similarity_keys:
        assert key in result["similarity_matrices"]
        assert result["similarity_matrices"][key].shape == (n_samples, n_samples)

    # Check that the fused matrix is symmetric
    assert np.allclose(result["fused_matrix"], result["fused_matrix"].T)

    # Check that the diagonal of the fused matrix is zero
    assert np.allclose(np.diag(result["fused_matrix"]), 0)
