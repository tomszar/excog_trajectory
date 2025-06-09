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
import matplotlib.gridspec as gridspec
from scipy.cluster import hierarchy

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


def plot_wgcna_clusters(
    wgcna_results: Dict[str, object],
    description_df: Optional[pd.DataFrame] = None,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize WGCNA clusters and their variables.

    Parameters
    ----------
    wgcna_results : Dict[str, object]
        Dictionary of WGCNA results as returned by run_wgcna()
    description_df : pd.DataFrame, optional
        DataFrame containing variable descriptions, with columns 'var' and 'description'
    figsize : tuple of int, default=(12, 10)
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
    """
    # Extract clusters from results
    clusters = wgcna_results['clusters']

    # Create a figure
    fig, ax = plt.subplots(figsize=figsize)

    # Hide axes
    ax.axis('off')

    # Create a text representation of the clusters
    cluster_text = "WGCNA Clusters\n" + "="*50 + "\n\n"

    # Sort clusters by size (descending)
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)

    # Add each cluster to the text
    for i, (cluster_id, variables) in enumerate(sorted_clusters):
        cluster_text += f"Cluster {i+1} (ID: {cluster_id}, Size: {len(variables)})\n"
        cluster_text += "-"*50 + "\n"

        # Add variables with descriptions if available
        for j, var in enumerate(sorted(variables)):
            if description_df is not None and 'var' in description_df.columns and 'description' in description_df.columns:
                # Try to find the variable in the description DataFrame
                desc_row = description_df[description_df['var'] == var]
                if not desc_row.empty:
                    description = desc_row['description'].values[0]
                    cluster_text += f"{j+1}. {var}: {description}\n"
                else:
                    cluster_text += f"{j+1}. {var}\n"
            else:
                cluster_text += f"{j+1}. {var}\n"

        cluster_text += "\n"

    # Add the text to the figure
    ax.text(0.05, 0.95, cluster_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', 
            family='monospace', wrap=True)

    # Save the figure if a path is provided
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig.savefig(os.path.join(save_path, 'wgcna_clusters.png'), 
                   bbox_inches='tight', dpi=300)

        # Also save as text file
        with open(os.path.join(save_path, 'wgcna_clusters.txt'), 'w') as f:
            f.write(cluster_text)

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
    save_path: Optional[str] = None,
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
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
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
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'exposure_correlation_matrix.png'), bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        return None

    return fig


def plot_missing_data_patterns(
    data: pd.DataFrame,
    variables: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 10),
    cluster_vars: bool = True,
    cmap: str = "viridis",
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Create a heatmap visualization of missing data patterns.

    This function creates a heatmap where each row represents an observation and each column
    represents a variable. Missing values are highlighted to visualize patterns of missingness.
    Variables can be clustered based on their missing data patterns.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the variables to analyze
    variables : List[str], optional
        List of variables to include in the visualization. If None, all variables are used.
    figsize : tuple of int, default=(12, 10)
        Figure size (width, height) in inches
    cluster_vars : bool, default=True
        Whether to cluster variables based on their missing data patterns
    cmap : str, default="viridis"
        Colormap to use for the heatmap
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
    dpi : int, default=300
        Resolution of the figure in dots per inch

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
    """
    # Select variables to visualize
    if variables is None:
        variables = data.columns.tolist()

    # Create a subset of the data with only the selected variables
    data_subset = data[variables].copy()

    # Create a binary matrix where 1 indicates missing value and 0 indicates non-missing
    missing_matrix = data_subset.isna().astype(int)

    # If clustering is requested, reorder variables based on their missing patterns
    if cluster_vars and len(variables) > 1:
        # Calculate distance matrix between variables based on their missing patterns
        # Use correlation as a measure of similarity in missing patterns
        var_corr = missing_matrix.corr()
        var_dist = 1 - np.abs(var_corr)

        # Perform hierarchical clustering
        linkage_matrix = hierarchy.linkage(hierarchy.distance.squareform(var_dist), method='average')

        # Reorder variables based on clustering
        dendro = hierarchy.dendrogram(linkage_matrix, no_plot=True)
        reordered_vars = [variables[i] for i in dendro['leaves']]

        # Reorder the missing matrix
        missing_matrix = missing_matrix[reordered_vars]

    # Create the figure
    fig = plt.figure(figsize=figsize)

    # Create a GridSpec layout with space for the heatmap and summary statistics
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4])

    # Create the main heatmap
    ax_heatmap = plt.subplot(gs[1, 0])

    # Plot the heatmap
    sns.heatmap(
        missing_matrix,
        cmap=cmap,
        cbar_kws={"shrink": 0.5, "label": "Missing (1) / Present (0)"},
        ax=ax_heatmap,
        yticklabels=False,  # Too many observations for individual labels
    )

    # Set title and labels
    ax_heatmap.set_title("Missing Data Patterns", fontsize=16)
    ax_heatmap.set_xlabel("Variables", fontsize=12)
    ax_heatmap.set_ylabel("Observations", fontsize=12)

    # Rotate x-axis labels for better readability
    plt.setp(ax_heatmap.get_xticklabels(), rotation=90)

    # Create a bar plot for the percentage of missing values per variable
    ax_bar = plt.subplot(gs[0, 0], sharex=ax_heatmap)

    # Calculate percentage of missing values for each variable
    missing_percentage = missing_matrix.mean() * 100

    # Plot the bar chart
    ax_bar.bar(range(len(missing_matrix.columns)), missing_percentage, color='steelblue')
    ax_bar.set_ylabel("% Missing", fontsize=10)
    ax_bar.set_ylim(0, 100)
    ax_bar.set_title("Percentage of Missing Values per Variable", fontsize=12)

    # Hide x-axis labels as they're shared with the heatmap
    ax_bar.set_xticklabels([])
    ax_bar.set_xticks([])

    # Create a histogram for the number of missing values per observation
    ax_hist = plt.subplot(gs[1, 1], sharey=ax_heatmap)

    # Calculate number of missing values for each observation
    missing_per_obs = missing_matrix.sum(axis=1)

    # Plot the histogram horizontally
    ax_hist.hist(missing_per_obs, bins=min(20, len(variables)), orientation='horizontal', color='steelblue')
    ax_hist.set_xlabel("Count", fontsize=10)
    ax_hist.set_title("Missing Values\nper Observation", fontsize=12)

    # Hide y-axis labels as they're shared with the heatmap
    ax_hist.set_yticklabels([])
    ax_hist.set_yticks([])

    # Add a text box with summary statistics
    ax_stats = plt.subplot(gs[0, 1])
    ax_stats.axis('off')

    # Calculate summary statistics
    total_cells = missing_matrix.size
    total_missing = missing_matrix.sum().sum()
    pct_missing = (total_missing / total_cells) * 100

    # Variables with any missing values
    vars_with_missing = sum(missing_matrix.sum() > 0)
    pct_vars_with_missing = (vars_with_missing / len(variables)) * 100

    # Observations with any missing values
    obs_with_missing = sum(missing_matrix.sum(axis=1) > 0)
    pct_obs_with_missing = (obs_with_missing / len(missing_matrix)) * 100

    # Create the text for the stats box
    stats_text = (
        f"Total Missing: {total_missing} ({pct_missing:.1f}%)\n"
        f"Variables with Missing: {vars_with_missing} ({pct_vars_with_missing:.1f}%)\n"
        f"Observations with Missing: {obs_with_missing} ({pct_obs_with_missing:.1f}%)"
    )

    # Add the text to the plot
    ax_stats.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=10, 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

    # Adjust layout
    plt.tight_layout()

    # Save the figure if a save path is provided
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'missing_data_patterns.png'), bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        return None

    return fig


def plot_pairwise_completeness(
    data: pd.DataFrame,
    variables: Optional[List[str]] = None,
    min_observations: int = 30,
    min_pct_observations: float = 0.3,
    figsize: Tuple[int, int] = (15, 12),
    cmap: str = "YlGnBu",
    cluster_vars: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Create a heatmap visualization of pairwise completeness between variables.

    This function creates a heatmap where each cell represents the number or percentage
    of complete observations for a pair of variables. Variables can be clustered based
    on their pairwise completeness patterns.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the variables to analyze
    variables : List[str], optional
        List of variables to include in the visualization. If None, all variables are used.
    min_observations : int, default=30
        Minimum number of complete observations required for a reliable correlation
    min_pct_observations : float, default=0.3
        Minimum percentage of total observations required for a reliable correlation
    figsize : tuple of int, default=(15, 12)
        Figure size (width, height) in inches
    cmap : str, default="YlGnBu"
        Colormap to use for the heatmap
    cluster_vars : bool, default=True
        Whether to cluster variables based on their pairwise completeness patterns
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
    dpi : int, default=300
        Resolution of the figure in dots per inch

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
    """
    # Select variables to visualize
    if variables is None:
        variables = data.columns.tolist()

    # Create a subset of the data with only the selected variables
    data_subset = data[variables].copy()

    # Calculate the total number of observations
    total_obs = len(data_subset)

    # Calculate the minimum number of observations based on percentage
    min_obs_from_pct = int(total_obs * min_pct_observations)

    # Use the maximum of the two thresholds
    effective_min_obs = max(min_observations, min_obs_from_pct)

    # Create matrices to store pairwise completeness information
    pairwise_complete_count = pd.DataFrame(index=variables, columns=variables)
    pairwise_complete_pct = pd.DataFrame(index=variables, columns=variables)
    pairwise_reliability = pd.DataFrame(index=variables, columns=variables)

    # Calculate pairwise completeness
    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            # For diagonal elements, use the non-missing count for the variable
            if i == j:
                complete_count = data_subset[var1].notna().sum()
            else:
                # For off-diagonal elements, count observations where both variables are non-missing
                complete_count = data_subset[[var1, var2]].dropna().shape[0]

            # Store the count and percentage
            pairwise_complete_count.loc[var1, var2] = complete_count
            pairwise_complete_pct.loc[var1, var2] = (complete_count / total_obs) * 100

            # Determine if the pair has enough observations for reliable correlation
            is_reliable = complete_count >= effective_min_obs
            pairwise_reliability.loc[var1, var2] = is_reliable

    # If clustering is requested, reorder variables based on their pairwise completeness patterns
    if cluster_vars and len(variables) > 1:
        # Use the percentage matrix for clustering
        # Convert to a distance matrix (higher completeness = lower distance)
        dist_matrix = 100 - pairwise_complete_pct

        # Ensure the distance matrix is symmetric
        dist_matrix = (dist_matrix + dist_matrix.T) / 2

        # Perform hierarchical clustering
        linkage_matrix = hierarchy.linkage(hierarchy.distance.squareform(dist_matrix), method='average')

        # Reorder variables based on clustering
        dendro = hierarchy.dendrogram(linkage_matrix, no_plot=True)
        reordered_vars = [variables[i] for i in dendro['leaves']]

        # Reorder the matrices
        pairwise_complete_count = pairwise_complete_count.loc[reordered_vars, reordered_vars]
        pairwise_complete_pct = pairwise_complete_pct.loc[reordered_vars, reordered_vars]
        pairwise_reliability = pairwise_reliability.loc[reordered_vars, reordered_vars]

        # Update the variables list
        variables = reordered_vars

    # Create the figure
    fig = plt.figure(figsize=figsize)

    # Create a GridSpec layout with space for multiple heatmaps
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[4, 1])

    # Create the count heatmap
    ax_count = plt.subplot(gs[0, 0])

    # Plot the count heatmap
    sns.heatmap(
        pairwise_complete_count,
        cmap=cmap,
        annot=True,
        fmt="d",
        ax=ax_count,
        cbar_kws={"shrink": 0.5, "label": "Complete Observations (count)"},
    )

    # Set title and labels
    ax_count.set_title("Pairwise Complete Observations (Count)", fontsize=14)
    ax_count.set_xlabel("Variables", fontsize=12)
    ax_count.set_ylabel("Variables", fontsize=12)

    # Rotate axis labels for better readability
    plt.setp(ax_count.get_xticklabels(), rotation=90)
    plt.setp(ax_count.get_yticklabels(), rotation=0)

    # Create the percentage heatmap
    ax_pct = plt.subplot(gs[0, 1])

    # Plot the percentage heatmap
    sns.heatmap(
        pairwise_complete_pct,
        cmap=cmap,
        annot=True,
        fmt=".1f",
        ax=ax_pct,
        cbar_kws={"shrink": 0.5, "label": "Complete Observations (%)"},
    )

    # Set title and labels
    ax_pct.set_title("Pairwise Complete Observations (%)", fontsize=14)
    ax_pct.set_xlabel("Variables", fontsize=12)
    ax_pct.set_ylabel("Variables", fontsize=12)

    # Rotate axis labels for better readability
    plt.setp(ax_pct.get_xticklabels(), rotation=90)
    plt.setp(ax_pct.get_yticklabels(), rotation=0)

    # Create the reliability heatmap
    ax_rel = plt.subplot(gs[1, :])

    # Calculate the number of reliable correlations per variable
    reliable_counts = pairwise_reliability.sum(axis=1)
    reliable_pcts = (reliable_counts / (len(variables) - 1)) * 100  # Exclude self-correlation

    # Create a DataFrame for the reliability summary
    reliability_df = pd.DataFrame({
        'variable': variables,
        'reliable_correlations': reliable_counts,
        'reliable_percentage': reliable_pcts
    })

    # Sort by reliable percentage
    reliability_df = reliability_df.sort_values('reliable_percentage', ascending=False)

    # Plot the reliability summary as a horizontal bar chart
    sns.barplot(
        x='reliable_percentage',
        y='variable',
        data=reliability_df,
        ax=ax_rel,
        palette='YlGnBu',
    )

    # Set title and labels
    ax_rel.set_title(f"Percentage of Reliable Correlations per Variable (min {effective_min_obs} observations)", fontsize=14)
    ax_rel.set_xlabel("Percentage of Reliable Correlations", fontsize=12)
    ax_rel.set_ylabel("Variables", fontsize=12)

    # Add text annotations with the actual counts
    for i, (_, row) in enumerate(reliability_df.iterrows()):
        ax_rel.text(
            row['reliable_percentage'] + 1,  # Offset for visibility
            i,
            f"{int(row['reliable_correlations'])}/{len(variables)-1}",
            va='center',
            fontsize=8,
        )

    # Adjust layout
    plt.tight_layout()

    # Save the figure if a save path is provided
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'pairwise_completeness.png'), bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        return None

    return fig
