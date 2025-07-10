"""
Visualization functions for exposomic and cognitive data.

This module provides functions for creating visualizations of NHANES data,
exposure-outcome relationships, and analysis results.
"""

from typing import Dict, List, Optional, Tuple

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting


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
        Number of cols in the grid of plots
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
        Number of cols in the grid of plots
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
    # Use all cols in data as exposure variables
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


def plot_plsr_scores(
        best_model,
        data_df: pd.DataFrame,
        cognitive_vars: List[str],
        output_dir: str,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = 'viridis',
        alpha: float = 0.7,
        dpi: int = 300,
) -> Dict[str, List[str]]:
    """
    Create scatter plots of selected pairs of columns of x_scores from a PLSR model.
    For n components, shows n/2 plots (rounded up) with pairs like (1,2), (3,4), etc.

    Parameters
    ----------
    best_model : PLSRegression
        The fitted PLSR model containing x_scores_
    data_df : pd.DataFrame
        DataFrame containing the cognitive variables for coloring the points
    cognitive_vars : List[str]
        List of cognitive variables to use for coloring the scatter plots
    output_dir : str
        Directory to save the plots
    figsize : tuple of int, default=(10, 8)
        Figure size (width, height) in inches
    cmap : str, default='viridis'
        Colormap to use for the scatter plots
    alpha : float, default=0.7
        Alpha value for the scatter points
    dpi : int, default=300
        Resolution of the figure in dots per inch

    Returns
    -------
    Dict[str, List[str]]
        Dictionary containing the paths of the saved plots
    """
    # Get the x_scores from the best_model
    x_scores = best_model.x_scores_

    # Get the number of components
    n_components = x_scores.shape[1]

    # Dictionary to store the paths of saved plots
    saved_plots = {"plots": []}

    # Create a scatter plot for each cognitive variable
    for i, cog_var in enumerate(cognitive_vars):
        # Calculate number of plots needed (only showing specific pairs)
        # For n_components, we'll show n_components/2 plots (rounded up)
        n_plots = int(np.ceil(n_components / 2))
        n_rows = int(np.ceil(n_plots / 2))  # 2 plots per row
        fig, axes = plt.subplots(n_rows, 2, figsize=(figsize[0] * 2, figsize[1] * n_rows))

        # Handle case where there's only one plot
        if n_plots == 1:
            axes = np.array([axes])

        axes = axes.flatten()

        # Plot specific pairs of components
        # For example: (1,2), (3,4), (5,6), etc.
        # If odd number of components, the last one pairs with the previous one
        scatter = None
        for plot_idx in range(n_plots):
            if plot_idx < len(axes):
                ax = axes[plot_idx]

                # Calculate which components to plot
                if plot_idx == n_plots - 1 and n_components % 2 == 1:
                    # For odd number of components, last plot is (n-2, n-1)
                    comp_i = n_components - 2
                    comp_j = n_components - 1
                else:
                    # Normal case: (0,1), (2,3), (4,5), etc.
                    comp_i = plot_idx * 2
                    comp_j = plot_idx * 2 + 1

                # Make sure we don't exceed the number of components
                if comp_j < n_components:
                    # Create a scatter plot with points colored by the cognitive variable
                    scatter = ax.scatter(
                        x_scores[:, comp_i],
                        x_scores[:, comp_j],
                        c=data_df[cog_var],
                        cmap=cmap,
                        alpha=alpha
                    )

                    # Add labels
                    ax.set_xlabel(f'Component {comp_i + 1}')
                    ax.set_ylabel(f'Component {comp_j + 1}')
                    ax.set_title(f'Components {comp_i + 1} vs {comp_j + 1}')

                    # Add a grid
                    ax.grid(True, linestyle='--', alpha=alpha)

        # Remove any unused subplots
        for idx in range(n_plots, len(axes)):
            fig.delaxes(axes[idx])

        # Add a colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.set_label(cog_var)

        # Add overall title
        fig.suptitle(f'PLSR Scores - Selected Component Pairs (Coded by {cog_var})', fontsize=16)

        # Save the plot
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        plot_path = os.path.join(output_dir, f"plsr_scores_scatter_{cog_var}.png")
        plt.savefig(plot_path, dpi=dpi)
        plt.close()

        # Add the path to the saved plots
        saved_plots["plots"].append(plot_path)

    # If no cognitive variables are available, create a simple scatter plot with specific pairs of components
    if not cognitive_vars:
        # Calculate number of plots needed (only showing specific pairs)
        # For n_components, we'll show n_components/2 plots (rounded up)
        n_plots = int(np.ceil(n_components / 2))
        n_rows = int(np.ceil(n_plots / 2))  # 2 plots per row
        fig, axes = plt.subplots(n_rows, 2, figsize=(figsize[0] * 2, figsize[1] * n_rows))

        # Handle case where there's only one plot
        if n_plots == 1:
            axes = np.array([axes])

        axes = axes.flatten()

        # Plot specific pairs of components
        # For example: (1,2), (3,4), (5,6), etc.
        # If odd number of components, the last one pairs with the previous one
        for plot_idx in range(n_plots):
            if plot_idx < len(axes):
                ax = axes[plot_idx]

                # Calculate which components to plot
                if plot_idx == n_plots - 1 and n_components % 2 == 1:
                    # For odd number of components, last plot is (n-2, n-1)
                    comp_i = n_components - 2
                    comp_j = n_components - 1
                else:
                    # Normal case: (0,1), (2,3), (4,5), etc.
                    comp_i = plot_idx * 2
                    comp_j = plot_idx * 2 + 1

                # Make sure we don't exceed the number of components
                if comp_j < n_components:
                    # Create a scatter plot
                    ax.scatter(
                        x_scores[:, comp_i],
                        x_scores[:, comp_j],
                        alpha=alpha
                    )

                    # Add labels
                    ax.set_xlabel(f'Component {comp_i + 1}')
                    ax.set_ylabel(f'Component {comp_j + 1}')
                    ax.set_title(f'Components {comp_i + 1} vs {comp_j + 1}')

                    # Add a grid
                    ax.grid(True, linestyle='--', alpha=alpha)

        # Remove any unused subplots
        for idx in range(n_plots, len(axes)):
            fig.delaxes(axes[idx])

        # Add overall title
        fig.suptitle('PLSR Scores - Selected Component Pairs', fontsize=16)

        # Save the plot
        plt.tight_layout(rect=[0, 0, 0.95, 0.95])
        plot_path = os.path.join(output_dir, "plsr_scores_scatter.png")
        plt.savefig(plot_path, dpi=dpi)
        plt.close()

        # Add the path to the saved plots
        saved_plots["plots"].append(plot_path)


    return saved_plots


def plot_cognitive_trajectory(
        x_scores: pd.DataFrame,
        obs_vect: pd.DataFrame,
        output_dir: str,
        figsize_2d: Tuple[int, int] = (20, 16),
        figsize_3d: Tuple[int, int] = (12, 10),
        dpi: int = 300,
) -> Dict[str, List[str]]:
    """
    Create 2D and 3D plots for cognitive trajectory by sex using obs_vect.

    Parameters
    ----------
    x_scores : pd.DataFrame
        DataFrame containing the PLSR scores for all samples
    obs_vect : pd.DataFrame
        DataFrame containing the values for each cognitive category for each sex
        using a linear regression that accommodates the inclusion of covariates
    output_dir : str
        Directory to save the plots
    figsize_2d : tuple of int, default=(20, 16)
        Figure size (width, height) in inches for 2D plots
    figsize_3d : tuple of int, default=(12, 10)
        Figure size (width, height) in inches for 3D plot
    dpi : int, default=300
        Resolution of the figure in dots per inch

    Returns
    -------
    Dict[str, List[str]]
        Dictionary containing the paths of the saved plots
    """
    # Dictionary to store the paths of saved plots
    saved_plots = {"plots": []}

    # Create 2D plots for cognitive trajectory by sex using obs_vect
    n_components = obs_vect.shape[1]
    n_plots = int(np.ceil(n_components / 2))
    n_rows = int(np.ceil(n_plots / 2))  # 2 plots per row

    # Define cognitive categories and sex labels
    cognitive_categories = ["Low", "Average", "High"]
    sex_labels = ["Female", "Male"]

    # Define colors and markers
    colors = {'High': 'green', 'Average': 'blue', 'Low': 'red'}
    markers = {'Female': 'o', 'Male': 's'}  # circle for females, square for males

    # Create 2D plots
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize_2d)

    # Handle case where there's only one plot
    if n_plots == 1:
        axes = np.array([axes])

    axes = axes.flatten()

    # Plot specific pairs of components
    for plot_idx in range(n_plots):
        if plot_idx < len(axes):
            ax = axes[plot_idx]

            # Calculate which components to plot
            if plot_idx == n_plots - 1 and n_components % 2 == 1:
                # For odd number of components, last plot is (n-2, n-1)
                comp_i = n_components - 2
                comp_j = n_components - 1
            else:
                # Normal case: (0,1), (2,3), (4,5), etc.
                comp_i = plot_idx * 2
                comp_j = plot_idx * 2 + 1

            # Make sure we don't exceed the number of components
            if comp_j < n_components:
                # Plot all points with low alpha for context
                ax.scatter(
                    x_scores.iloc[:, comp_i],
                    x_scores.iloc[:, comp_j],
                    color='lightgray',
                    alpha=0.1
                )

                # For each sex
                for sex_idx, sex in enumerate(sex_labels):
                    # Get indices for this sex in obs_vect
                    # Female: rows 0-2, Male: rows 3-5
                    start_idx = sex_idx * 3
                    end_idx = start_idx + 3

                    # Store points for connecting lines
                    line_points_x = []
                    line_points_y = []

                    # For each cognitive category
                    for cat_idx, cat in enumerate(cognitive_categories):
                        # Get the corresponding row index in obs_vect
                        row_idx = start_idx + cat_idx

                        # Get the x and y coordinates from obs_vect
                        x_coord = obs_vect.iloc[row_idx, comp_i]
                        y_coord = obs_vect.iloc[row_idx, comp_j]

                        # Store coordinates for connecting lines
                        line_points_x.append(x_coord)
                        line_points_y.append(y_coord)

                        # Plot point with larger marker and label
                        ax.scatter(
                            x_coord,
                            y_coord,
                            color=colors[cat],
                            marker=markers[sex],
                            s=150,  # Larger size for visibility
                            alpha=1.0,
                            edgecolors='black',
                            linewidths=1.5,
                            label=f'{cat} - {sex}' if plot_idx == 0 else ""
                        )

                    # Connect the points with lines
                    line_style = '--' if sex == 'Female' else '-'
                    ax.plot(
                        line_points_x,
                        line_points_y,
                        color='black',
                        linestyle=line_style,
                        alpha=0.7,
                        label=f'{sex} Trajectory' if plot_idx == 0 else ""
                    )

                # Add labels
                ax.set_xlabel(f'Component {comp_i + 1}')
                ax.set_ylabel(f'Component {comp_j + 1}')
                ax.set_title(f'Components {comp_i + 1} vs {comp_j + 1}')

                # Add a grid
                ax.grid(True, linestyle='--', alpha=0.3)

                # Set x and y axis limits
                ax.set_xlim([-1.5, 1.5])
                ax.set_ylim([-1.5, 1.5])

                # Add legend only to the first subplot
                if plot_idx == 0:
                    ax.legend(loc='best', fontsize='small')

    # Remove any unused subplots
    for idx in range(n_plots, len(axes)):
        fig.delaxes(axes[idx])

    # Add overall title
    fig.suptitle('Cognitive Trajectory by Sex (Selected Component Pairs)', fontsize=16)

    # Save the plot
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plot_path = os.path.join(output_dir, "cognitive_trajectory_2d.png")
    plt.savefig(plot_path, dpi=dpi)
    plt.close()

    # Add the path to the saved plots
    saved_plots["plots"].append(plot_path)

    # Create 3D plot if we have at least 3 components
    if n_components >= 3:
        fig = plt.figure(figsize=figsize_3d)
        ax = fig.add_subplot(111, projection='3d')

        # Plot all points with low alpha for context
        ax.scatter(
            x_scores.iloc[:, 0],
            x_scores.iloc[:, 1],
            x_scores.iloc[:, 2],
            color='lightgray',
            alpha=0.1
        )

        # For each sex
        for sex_idx, sex in enumerate(sex_labels):
            # Get indices for this sex in obs_vect
            # Female: rows 0-2, Male: rows 3-5
            start_idx = sex_idx * 3
            end_idx = start_idx + 3

            # Store points for connecting lines
            line_points_x = []
            line_points_y = []
            line_points_z = []

            # For each cognitive category
            for cat_idx, cat in enumerate(cognitive_categories):
                # Get the corresponding row index in obs_vect
                row_idx = start_idx + cat_idx

                # Get the x, y, and z coordinates from obs_vect
                x_coord = obs_vect.iloc[row_idx, 0]
                y_coord = obs_vect.iloc[row_idx, 1]
                z_coord = obs_vect.iloc[row_idx, 2]

                # Store coordinates for connecting lines
                line_points_x.append(x_coord)
                line_points_y.append(y_coord)
                line_points_z.append(z_coord)

                # Plot point with larger marker and label
                ax.scatter(
                    x_coord,
                    y_coord,
                    z_coord,
                    color=colors[cat],
                    marker=markers[sex],
                    s=150,  # Larger size for visibility
                    alpha=1.0,
                    edgecolors='black',
                    linewidths=1.5,
                    label=f'{cat} - {sex}'
                )

            # Connect the points with lines
            line_style = '--' if sex == 'Female' else '-'
            ax.plot(
                line_points_x,
                line_points_y,
                line_points_z,
                color='black',
                linestyle=line_style,
                alpha=0.7,
                label=f'{sex} Trajectory'
            )

        # Add labels
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        ax.set_title('Cognitive Trajectory by Sex (3D)')

        # Add a legend
        ax.legend(loc='best')

        # Set axis limits
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])

        # Save the plot
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "cognitive_trajectory_3d.png")
        plt.savefig(plot_path, dpi=dpi)
        plt.close()

        # Add the path to the saved plots
        saved_plots["plots"].append(plot_path)

    return saved_plots
