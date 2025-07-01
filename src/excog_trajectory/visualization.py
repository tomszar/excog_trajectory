"""
Visualization functions for exposomic and cognitive data.

This module provides functions for creating visualizations of NHANES data,
exposure-outcome relationships, and analysis results.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
        figsize: Tuple[int, int] = (20, 20),
        cmap: str = "seismic",
        fname: Optional[str] = None,
        dpi: int = 300,
) -> plt.Figure:
    """
    Create a correlation matrix between all exposure variables.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing exposure variables
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
    # Use all columns in data as exposure variables
    exposure_vars = list(data.columns)

    if not exposure_vars:
        print("No exposure variables found in the data")
        fig, ax = plt.subplots(figsize=figsize)
        return fig

    # Create a dataframe with only exposure variables
    exposure_data = data[exposure_vars].copy()

    # Calculate correlation matrix
    corr_matrix = exposure_data.corr()

    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        corr_matrix,
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

    # Set tick labels and ensure all are displayed
    ax.set_xticks(np.arange(len(exposure_vars)) + 0.5)
    ax.set_yticks(np.arange(len(exposure_vars)) + 0.5)
    ax.set_xticklabels(exposure_vars, rotation=90, fontsize=4)
    ax.set_yticklabels(exposure_vars, rotation=0, fontsize=4)

    # Adjust layout
    plt.tight_layout()

    # Save the figure if a save path is provided
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        return None

    return fig
