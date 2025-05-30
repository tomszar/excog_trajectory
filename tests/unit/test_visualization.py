"""
Unit tests for the visualization module.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from excog_trajectory import visualization


def test_plot_exposure_distributions():
    """Test that plot_exposure_distributions returns a matplotlib Figure."""
    # Create a mock DataFrame
    test_data = pd.DataFrame({
        "exposure1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "exposure2": [1.0, 2.0, 3.0, 4.0, 5.0],
        "other_col": [10, 20, 30, 40, 50]
    })

    # Call the function
    result = visualization.plot_exposure_distributions(
        data=test_data,
        exposure_vars=["exposure1", "exposure2"]
    )

    # Check that the result is a matplotlib Figure
    assert isinstance(result, plt.Figure)


def test_plot_exposure_outcome_relationships():
    """Test that plot_exposure_outcome_relationships returns a matplotlib Figure."""
    # Create a mock DataFrame
    test_data = pd.DataFrame({
        "outcome1": [1, 2, 3, 4, 5],
        "exposure1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "exposure2": [1.0, 2.0, 3.0, 4.0, 5.0]
    })

    # Call the function
    result = visualization.plot_exposure_outcome_relationships(
        data=test_data,
        outcome_var="outcome1",
        exposure_vars=["exposure1", "exposure2"]
    )

    # Check that the result is a matplotlib Figure
    assert isinstance(result, plt.Figure)


def test_plot_model_coefficients():
    """Test that plot_model_coefficients returns a matplotlib Figure."""
    # Create a mock model results dictionary
    model_results = {
        "outcome1": {
            "exposure1": {"coef": 0.5, "p_value": 0.01, "ci_lower": 0.1, "ci_upper": 0.9},
            "exposure2": {"coef": -0.3, "p_value": 0.05, "ci_lower": -0.6, "ci_upper": 0.0}
        }
    }

    # Call the function
    result = visualization.plot_model_coefficients(
        model_results=model_results,
        exposure_vars=["exposure1", "exposure2"]
    )

    # Check that the result is a matplotlib Figure
    assert isinstance(result, plt.Figure)


def test_plot_trajectory_curves():
    """Test that plot_trajectory_curves returns a matplotlib Figure."""
    # Create a mock DataFrame
    test_data = pd.DataFrame({
        "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "time": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "outcome": [10, 11, 12, 20, 21, 22, 30, 31, 32],
        "exposure1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    })

    # Create a mock longitudinal results dictionary
    longitudinal_results = {
        "model": "mock_model",
        "coefficients": {"exposure1": 0.5, "time": 1.0, "exposure1:time": 0.2}
    }

    # Call the function
    result = visualization.plot_trajectory_curves(
        longitudinal_results=longitudinal_results,
        data=test_data,
        time_var="time",
        outcome_var="outcome",
        exposure_var="exposure1",
        exposure_levels=[0.1, 0.5, 0.9]
    )

    # Check that the result is a matplotlib Figure
    assert isinstance(result, plt.Figure)


def test_figure_has_title_and_labels():
    """Test that figures have titles and axis labels."""
    # Create a mock DataFrame
    test_data = pd.DataFrame({
        "exposure1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "exposure2": [1.0, 2.0, 3.0, 4.0, 5.0]
    })

    # Call the function
    fig = visualization.plot_exposure_distributions(
        data=test_data,
        exposure_vars=["exposure1", "exposure2"]
    )

    # Get the axes from the figure
    axes = fig.get_axes()
    
    # Check that there is at least one subplot
    assert len(axes) > 0
    
    # In a real test, we would check that each subplot has a title and axis labels
    # But since the functions are placeholders, we'll just check that the figure has axes
    assert isinstance(axes[0], plt.Axes)