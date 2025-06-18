"""
Visualization functions for exposomic and cognitive data.

This module provides functions for creating visualizations of NHANES data,
exposure-outcome relationships, and analysis results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Union

from matplotlib.figure import Figure


def plot_distributions(
    data: pd.DataFrame,
    vars: List[str],
    n_cols: int = 3,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None,
) -> Figure | None:
    """
    Create histograms or density plots of variable distributions.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing variables to plot
    vars : list of str
        List of variables to plot
    n_cols : int, default=3
        Number of columns in the grid of plots
    figsize : tuple of int, default=(15, 10)
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.

    Returns
    -------
    matplotlib.figure.Figure or None
        The created figure object
    """
    # Placeholder for actual implementation
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    axes.hist(data[vars])
    if save_path is not None:
        fig.savefig(save_path + '/distributions.png', bbox_inches='tight')
        return None
    else:
        return fig


def plot_exposure_outcome_relationships(
    data: pd.DataFrame,
    outcome_var: str,
    exposure_vars: List[str],
    n_cols: int = 3,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create scatter plots of relationships between exposures and cognitive outcomes.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing exposure and outcome variables
    outcome_var : str
        Cognitive outcome variable to plot
    exposure_vars : list of str
        List of exposure variables to plot against the outcome
    n_cols : int, default=3
        Number of columns in the grid of plots
    figsize : tuple of int, default=(15, 10)
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
    """
    # Placeholder for actual implementation
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    return fig


def plot_model_coefficients(
    model_results: Dict[str, Dict[str, object]],
    exposure_vars: List[str],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create forest plots of model coefficients for exposure variables.

    Parameters
    ----------
    model_results : Dict[str, Dict[str, object]]
        Dictionary of model results as returned by run_linear_models()
    exposure_vars : list of str
        List of exposure variables to include in the plot
    figsize : tuple of int, default=(12, 8)
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
    """
    # Placeholder for actual implementation
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    return fig


def plot_trajectory_curves(
    longitudinal_results: Dict[str, object],
    data: pd.DataFrame,
    time_var: str,
    outcome_var: str,
    exposure_var: str,
    exposure_levels: List[float],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot predicted cognitive trajectories at different exposure levels.

    Parameters
    ----------
    longitudinal_results : Dict[str, object]
        Dictionary of longitudinal model results as returned by run_longitudinal_analysis()
    data : pd.DataFrame
        DataFrame containing the original data
    time_var : str
        Variable indicating time point
    outcome_var : str
        Cognitive outcome variable
    exposure_var : str
        Exposure variable of interest
    exposure_levels : list of float
        Exposure levels at which to plot trajectories
    figsize : tuple of int, default=(10, 6)
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
    """
    # Placeholder for actual implementation
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    return fig


def plot_exposure_correlation_matrix(
    data: pd.DataFrame,
    description_df: pd.DataFrame,
    figsize: Tuple[int, int] = (20, 20),
    cmap: str = "seismic",
    fname: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Create a correlation matrix between all exposure variables, ordered by category.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing exposure variables
    description_df : pd.DataFrame
        DataFrame containing variable descriptions with 'var' and 'category' columns
    figsize : tuple of int, default=(20, 20)
        Figure size (width, height) in inches
    cmap : str, default="seismic"
        Colormap to use for the heatmap
    fname : str, optional
        File name and path to save the figure. If None, the figure is not saved.
    dpi : int, default=300
        Resolution of the figure in dots per inch

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
    """
    # Get all exposure variables from the data
    exposure_vars = [col for col in data.columns 
                    if col in description_df['var'].values]

    if not exposure_vars:
        print("No exposure variables found in the data")
        fig, ax = plt.subplots(figsize=figsize)
        return fig

    # Create a dataframe with only exposure variables
    exposure_data = data[exposure_vars].copy()

    # Calculate correlation matrix
    corr_matrix = exposure_data.corr()

    # Get category for each variable
    var_categories = {}
    for var in exposure_vars:
        if var in description_df['var'].values:
            category = description_df.loc[description_df['var'] == var, 'category'].iloc[0]
            var_categories[var] = category
        else:
            var_categories[var] = "unknown"

    # Create a DataFrame with variable names and categories
    var_info = pd.DataFrame({
        'var': list(var_categories.keys()),
        'category': list(var_categories.values())
    })

    # Sort variables by category
    var_info_sorted = var_info.sort_values('category')
    sorted_vars = var_info_sorted['var'].tolist()
    sorted_categories = var_info_sorted['category'].tolist()

    # Reorder correlation matrix
    corr_matrix_sorted = corr_matrix.loc[sorted_vars, sorted_vars]

    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        corr_matrix_sorted,
        cmap=cmap,
        annot=False,  # Too many variables for annotations
        square=True,
        ax=ax,
        vmin=-1,
        vmax=1,
        cbar_kws={"shrink": 0.5},  # Reduce the size of the colorbar
    )

    # Set title and labels
    ax.set_title("Correlation Matrix of Exposure Variables", fontsize=16)
    ax.set_xlabel("Exposure Variables", fontsize=12)
    ax.set_ylabel("Exposure Variables", fontsize=12)

    # Create tick labels with category information
    category_var_labels = [f"{cat}:{var}" for cat, var in zip(sorted_categories, sorted_vars)]

    # Set tick labels and ensure all are displayed
    ax.set_xticks(np.arange(len(sorted_vars)) + 0.5)
    ax.set_yticks(np.arange(len(sorted_vars)) + 0.5)
    ax.set_xticklabels(category_var_labels, rotation=90, fontsize=4)
    ax.set_yticklabels(category_var_labels, rotation=0, fontsize=4)

    # Draw category separators
    prev_category = None
    for i, category in enumerate(sorted_categories):
        if category != prev_category:
            # Draw horizontal and vertical lines to separate categories
            ax.axhline(y=i, color='black', linewidth=0.5)
            ax.axvline(x=i, color='black', linewidth=0.5)
            prev_category = category

    # Adjust layout
    plt.tight_layout()

    # Save the figure if a save path is provided
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        return None

    return fig
