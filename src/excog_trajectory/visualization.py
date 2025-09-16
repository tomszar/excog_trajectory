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
        n_cols: int = 2,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None,
        split_by_sex: bool = False,
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
    split_by_sex : bool, default=False
        If True, split distributions by sex categories (0 for females and 1 for males)
        using the RIAGENDR_1.0 column.

    Returns
    -------
    matplotlib.figure.Figure or None
        The created figure object
    """
    # Calculate number of rows needed based on number of variables and columns
    n_vars = len(vars)
    n_rows = int(np.ceil(n_vars / n_cols))

    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Ensure axes is always a 2D array for consistent indexing
    if n_vars == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    # Flatten axes for easier iteration
    axes_flat = axes.flatten()

    # Plot each variable
    for i, var in enumerate(vars):
        if i < len(axes_flat):
            ax = axes_flat[i]

            if split_by_sex and 'RIAGENDR_1.0' in data.columns:
                # Split by sex
                females = data[data['RIAGENDR_1.0'] == 0]
                males = data[data['RIAGENDR_1.0'] == 1]

                # Plot histograms for females and males
                if not females.empty and var in females.columns:
                    ax.hist(females[var], alpha=0.5, label='Female')

                if not males.empty and var in males.columns:
                    ax.hist(males[var], alpha=0.5, label='Male')

                ax.legend()
            else:
                # Plot histogram without splitting
                if var in data.columns:
                    ax.hist(data[var])

            # Set title and labels
            ax.set_title(var)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')

    # Hide any unused subplots
    for j in range(n_vars, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Adjust layout
    plt.tight_layout()

    # Save figure if save_path is provided
    if save_path is not None:
        fig.savefig(os.path.join(save_path, 'distributions.png'), bbox_inches='tight')
        return None
    else:
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
        plt.tight_layout()
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


def plot_vip_x_scores(
    vip_x_scores: pd.DataFrame,
    output_dir: str,
    figsize: Tuple[int, int] = (12, 10),
    top_n: int = 20,
    cmap: str = 'viridis',
    dpi: int = 300,
) -> Dict[str, List[str]]:
    """
    Create bar plots to illustrate the VIP-X scores per variable for each component and the cumulative scores.

    Parameters
    ----------
    vip_x_scores : pd.DataFrame
        DataFrame containing VIP-X scores for each predictor variable per component and cumulative
    output_dir : str
        Directory to save the plots
    figsize : tuple of int, default=(12, 10)
        Figure size (width, height) in inches
    top_n : int, default=20
        Number of top variables to show in each plot
    cmap : str, default='viridis'
        Colormap to use for the bar plots
    dpi : int, default=300
        Resolution of the figure in dots per inch

    Returns
    -------
    Dict[str, List[str]]
        Dictionary containing the paths of the saved plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to store the paths of saved plots
    saved_plots = {"plots": []}

    # Get the number of components
    n_components = len(vip_x_scores.columns) - 1  # Subtract 1 for the 'Cumulative' column

    # Create a color map
    colors = plt.cm.get_cmap(cmap, n_components + 1)

    # Plot cumulative VIP-X scores
    plt.figure(figsize=figsize)

    # Sort by cumulative VIP-X scores and get top_n variables
    top_vars_cumulative = vip_x_scores.sort_values('Cumulative', ascending=False).head(top_n)

    # Create horizontal bar plot
    bars = plt.barh(top_vars_cumulative.index, top_vars_cumulative['Cumulative'], color=colors(n_components))

    # Add labels and title
    plt.xlabel('VIP-X Score')
    plt.ylabel('Variable')
    plt.title(f'Top {top_n} Variables by Cumulative VIP-X Score')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Add a reference line at VIP-X = 1
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    cumulative_plot_path = os.path.join(output_dir, 'vip_x_cumulative.png')
    plt.savefig(cumulative_plot_path, dpi=dpi)
    plt.close()

    saved_plots["plots"].append(cumulative_plot_path)

    # Plot VIP-X scores for each component
    for comp in range(n_components):
        comp_name = f'Component_{comp+1}'

        plt.figure(figsize=figsize)

        # Sort by component VIP-X scores and get top_n variables
        top_vars_comp = vip_x_scores.sort_values(comp_name, ascending=False).head(top_n)

        # Create horizontal bar plot
        bars = plt.barh(top_vars_comp.index, top_vars_comp[comp_name], color=colors(comp))

        # Add labels and title
        plt.xlabel('VIP-X Score')
        plt.ylabel('Variable')
        plt.title(f'Top {top_n} Variables by VIP-X Score for {comp_name}')
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        # Add a reference line at VIP-X = 1
        plt.axvline(x=1, color='red', linestyle='--', alpha=0.7)

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        comp_plot_path = os.path.join(output_dir, f'vip_x_{comp_name}.png')
        plt.savefig(comp_plot_path, dpi=dpi)
        plt.close()

        saved_plots["plots"].append(comp_plot_path)

    # Create a combined plot showing all components and cumulative scores
    # This is useful for comparing the importance of variables across components

    # Get the top variables based on cumulative scores
    top_vars_overall = vip_x_scores.sort_values('Cumulative', ascending=False).head(top_n)

    # Create a figure with a larger width to accommodate the legend
    plt.figure(figsize=(figsize[0] + 2, figsize[1]))

    # Plot each component and cumulative scores
    bar_width = 0.8 / (n_components + 1)  # Width of each bar

    for i, col in enumerate(vip_x_scores.columns):
        # Calculate position for this set of bars
        pos = np.arange(len(top_vars_overall.index)) - 0.4 + i * bar_width

        # Create bars
        plt.barh(pos, top_vars_overall[col], height=bar_width, 
                label=col, color=colors(i), alpha=0.7)

    # Add labels and title
    plt.xlabel('VIP-X Score')
    plt.yticks(np.arange(len(top_vars_overall.index)), top_vars_overall.index)
    plt.title(f'Top {top_n} Variables by VIP-X Score Across All Components')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Add a reference line at VIP-X = 1
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.7)

    # Add legend
    plt.legend(loc='best')

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    combined_plot_path = os.path.join(output_dir, 'vip_x_combined.png')
    plt.savefig(combined_plot_path, dpi=dpi)
    plt.close()

    saved_plots["plots"].append(combined_plot_path)

    return saved_plots


def plot_plsr_biplot(
    model,
    X: pd.DataFrame,
    outcome_names: List[str],
    output_dir: str,
    vip_series: Optional[pd.Series] = None,
    filename: str = "plsr_biplot_all_components.png",
    components: Optional[Tuple[int, int]] = None,
    top_n_labels: int = 10,
    exposures_per_plot: int = 25,
    vip_scores: Optional[pd.DataFrame] = None,
    dpi: int = 300,
) -> str:
    """
    Create PLSR biplots showing, for each component pair, the relationship between:
    - Sample scores (gray points)
    - Exposure variables (blue arrows, colored by VIP intensity if provided)
    - Outcome variables (red arrows)

    Behavior
    --------
    - If `components` is None (default): iterate internally over all pairs (1,2), (3,4), ...
      and render them as subplots in a single figure. The figure is saved to `filename`.
    - If `components` is a tuple of two 1-indexed component numbers: render only that pair
      as a single-axes figure and save to `filename`.

    VIP coloring
    ------------
    - If `vip_scores` (DataFrame) is provided, per-plot coloring uses only VIPs for the displayed
      LVs. For a plot LV i vs LV j, exposures are colored by a per-plot VIP metric computed as
      max(VIP_i, VIP_j). Each subplot gets its own colorbar with the corresponding VIP scale.
    - If `vip_scores` is None but `vip_series` is provided, use that single Series for coloring
      across all subplots (backward-compatible). A colorbar is added to each subplot with the same scale.

    Parameters
    ----------
    model : PLSRegression
        Fitted PLSRegression model with x_scores_, x_loadings_, y_loadings_.
    X : pd.DataFrame
        Predictor matrix used to fit the model (for exposure names and ordering).
    outcome_names : List[str]
        Names of outcome variables corresponding to y_loadings_.
    output_dir : str
        Directory to save the plot.
    vip_series : Optional[pd.Series]
        Deprecated in favor of `vip_scores`. If provided (and `vip_scores` is None), exposure arrows
        are colored by this Series using the Blues colormap.
    filename : str, default="plsr_biplot_all_components.png"
        Output filename for the saved figure.
    components : Optional[Tuple[int, int]], default=None
        Which components to plot (1-indexed). If None, plot all pairs in one figure.
    top_n_labels : int, default=10
        Number of exposure arrows to label (highest VIP in that subplot). All outcomes are labeled.
    exposures_per_plot : int, default=25
        Maximum number of exposure arrows to draw per subplot (most relevant by VIP for that subplot).
    vip_scores : Optional[pd.DataFrame]
        VIP table as returned by analysis.calculate_vip_x_scores: columns 'Component_1', ..., 'Cumulative'.
        Index must be exposure names matching X columns.
    dpi : int, default=300
        Resolution of the saved figure.

    Returns
    -------
    str
        Path to the saved figure.
    """
    # Validate available components
    n_comp_scores = getattr(model, "x_scores_", np.empty((0, 0))).shape[1]
    n_comp_xload = getattr(model, "x_loadings_", np.empty((0, 0))).shape[1]
    n_comp_yload = getattr(model, "y_loadings_", np.empty((0, 0))).shape[1] if hasattr(model, "y_loadings_") else 0
    max_avail = max(0, min(n_comp_scores, n_comp_xload, n_comp_yload if n_comp_yload > 0 else n_comp_scores))
    if max_avail < 2:
        # Need at least two components for a biplot
        return ""

    exp_names = list(X.columns)

    # Helper to get per-plot VIP values and colors for the specific pair
    def _pair_vip_and_colors(comp_i_zero: int, comp_j_zero: int):
        cmap = plt.cm.Blues
        # Prefer the new combined-subset VIP over the plotted pair (LV_i, LV_j)
        try:
            # Lazy import to avoid circular dependencies at module import time
            from excog_trajectory import analysis as _analysis
            comb = _analysis.calculate_vip_x_scores(
                model, X, components=[comp_i_zero + 1, comp_j_zero + 1]
            )
            # comb is a single-column DataFrame; align to exposure names
            aligned = comb.reindex(exp_names)
            vals = aligned.iloc[:, 0]
        except Exception:
            # Backward compatibility and fallbacks
            if vip_scores is not None:
                aligned = vip_scores.reindex(exp_names)
                ci = f"Component_{comp_i_zero + 1}"
                cj = f"Component_{comp_j_zero + 1}"
                vals_i = aligned[ci] if ci in aligned.columns else None
                vals_j = aligned[cj] if cj in aligned.columns else None
                if vals_i is None and vals_j is None:
                    vals = (
                        aligned["Cumulative"]
                        if "Cumulative" in aligned.columns
                        else pd.Series(1.0, index=exp_names)
                    )
                elif vals_i is None:
                    vals = vals_j
                elif vals_j is None:
                    vals = vals_i
                else:
                    # Previous behavior: max across the two components
                    vals = pd.concat([vals_i, vals_j], axis=1).max(axis=1)
            elif vip_series is not None:
                vals = pd.Series(vip_series, index=exp_names)
            else:
                vals = pd.Series(1.0, index=exp_names)

        # Handle missing values
        if vals.isna().any():
            fill_val = float(np.nanmin(vals.values)) if np.any(~np.isnan(vals.values)) else 0.0
            vals = vals.fillna(fill_val)

        # Determine which exposures to draw (top by VIP for this pair)
        k = min(exposures_per_plot, len(exp_names))
        order = np.argsort(-vals.values)
        top_idx = list(order[:k])

        # Normalize colors based on the values of the drawn exposures
        if len(top_idx) > 0:
            vmin = float(vals.values[top_idx].min())
            vmax = float(vals.values[top_idx].max())
            if np.isclose(vmin, vmax):
                vmin = vmax - 1e-6
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            colors = {idx: cmap(norm(vals.iloc[idx])) for idx in top_idx}
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
        else:
            colors = {}
            sm = None

        # Determine which of the drawn exposures to label
        label_count = min(top_n_labels, len(top_idx))
        label_idx = set(top_idx[:label_count])

        return vals, top_idx, colors, sm

    def _draw_pair(ax, comp_i_zero: int, comp_j_zero: int):
        # Slice scores and loadings for the given pair (0-indexed)
        scores = model.x_scores_[:, [comp_i_zero, comp_j_zero]]
        x_load = model.x_loadings_[:, [comp_i_zero, comp_j_zero]]
        y_load = model.y_loadings_[:, [comp_i_zero, comp_j_zero]] if hasattr(model, "y_loadings_") else None

        # VIP for this pair
        vals, draw_indices, color_map, sm = _pair_vip_and_colors(comp_i_zero, comp_j_zero)

        # Scale arrows to fit the score space for this pair
        max_score = float(np.max(np.abs(scores))) if scores.size > 0 else 1.0
        mats = [x_load]
        if y_load is not None:
            mats.append(y_load)
        max_loading_vec = max(
            1e-12 + max(
                float(np.max(np.sqrt(np.sum(m**2, axis=1)))) if m.size > 0 else 0.0
                for m in mats
            ),
            1e-6,
        )
        arrow_scale = (max_score * 0.9) / max_loading_vec

        # Scatter sample scores
        ax.scatter(
            scores[:, 0],
            scores[:, 1],
            color="lightgray",
            alpha=0.1,
            s=10,
            label="Samples"
        )

        # Draw exposure arrows and labels (only top exposures for this pair)
        for idx in draw_indices:
            name = exp_names[idx]
            dx, dy = arrow_scale * x_load[idx, 0], arrow_scale * x_load[idx, 1]
            ax.arrow(
                0,
                0,
                dx,
                dy,
                color=color_map.get(idx, "#1f77b4"),
                alpha=0.9,
                width=0.001,
                head_width=0.1,
                head_length=0.1,
                length_includes_head=True,
            )
        # Label only the top_n among drawn exposures
        label_count = min(top_n_labels, len(draw_indices))
        for idx in draw_indices[:label_count]:
            name = exp_names[idx]
            dx, dy = arrow_scale * x_load[idx, 0], arrow_scale * x_load[idx, 1]
            ax.text(
                dx * 1.05,
                dy * 1.05,
                name,
                fontsize=9,
                ha="center",
                va="center",
                color=color_map.get(idx, "#1f77b4"),
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )

        # Draw outcome arrows and labels
        if y_load is not None and len(outcome_names) == y_load.shape[0]:
            for jdx, oname in enumerate(outcome_names):
                dx, dy = arrow_scale * y_load[jdx, 0], arrow_scale * y_load[jdx, 1]
                ax.arrow(
                    0,
                    0,
                    dx,
                    dy,
                    fc="red",
                    ec="red",
                    alpha=0.9,
                    width=0.001,
                    head_width=0.1,
                    head_length=0.1,
                    length_includes_head=True,
                )
                ax.text(
                    dx * 1.05,
                    dy * 1.05,
                    oname,
                    fontsize=10,
                    ha="center",
                    va="center",
                    color="darkred",
                    fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )

        # Axes through origin and labels
        ax.axhline(0, color="black", linewidth=0.7, alpha=0.7)
        ax.axvline(0, color="black", linewidth=0.7, alpha=0.7)
        ax.set_xlabel(f"LV{comp_i_zero + 1}")
        ax.set_ylabel(f"LV{comp_j_zero + 1}")
        ax.set_title(f"Exposures and outcomes in latent space (LV{comp_i_zero + 1} vs LV{comp_j_zero + 1})")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")

        # Add per-axes colorbar if VIP available
        if sm is not None:
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f"VIP (LV{comp_i_zero + 1} + LV{comp_j_zero + 1})")

    os.makedirs(output_dir, exist_ok=True)

    # If a specific pair is requested, render a single-axes figure
    if components is not None:
        comp_i = max(0, min(max_avail - 1, components[0] - 1))
        comp_j = max(0, min(max_avail - 1, components[1] - 1))
        if comp_i == comp_j:
            # Ensure two distinct components
            comp_j = min(max_avail - 1, comp_i + 1)
        fig, ax = plt.subplots(figsize=(8, 8))
        _draw_pair(ax, comp_i, comp_j)
        out_path = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(out_path, dpi=dpi)
        plt.close(fig)
        return out_path

    # Otherwise, render all pairs as subplots in a single figure
    n_components = max_avail
    n_plots = int(np.ceil(n_components / 2))
    n_rows = int(np.ceil(n_plots / 2))  # 2 plots per row
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))

    # Flatten axes for easy indexing
    axes = np.atleast_1d(axes).flatten()

    for plot_idx in range(n_plots):
        if plot_idx == n_plots - 1 and n_components % 2 == 1:
            # Odd number of components: last plot is (n-1, n)
            comp_i = n_components - 2
            comp_j = n_components - 1
        else:
            comp_i = plot_idx * 2
            comp_j = plot_idx * 2 + 1
        if plot_idx < len(axes):
            _draw_pair(axes[plot_idx], comp_i, comp_j)

    # Remove any unused subplots
    for idx in range(n_plots, len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle("PLSR biplots across component pairs", fontsize=14)
    out_path = os.path.join(output_dir, filename)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_cognitive_trajectory(
        x_scores: pd.DataFrame,
        obs_vect: pd.DataFrame,
        output_dir: str,
        filename: str = "cognitive_trajectory",
        dpi: int = 300,
        factor: int = 3,
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
    filename :  str, default="cognitive_trajectory"
        Name of the output file
    dpi : int, default=300
        Resolution of the figure in dots per inch
    factor : int, default=3
        Factor to multiply the x and y scores for better visualization

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
    n_cols = 2

    # Define cognitive categories and sex labels
    cognitive_categories = ["Low", "Average", "High"]
    sex_labels = ["Female", "Male"]

    # Define colors and markers
    colors = {'High': 'green', 'Average': 'blue', 'Low': 'red'}
    markers = {'Female': 'o', 'Male': 's'}  # circle for females, square for males

    # Create 2D plots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))

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
                    alpha=0.1,
                    s=10,
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
                        x_coord = obs_vect.iloc[row_idx, comp_i] * factor
                        y_coord = obs_vect.iloc[row_idx, comp_j] * factor

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
                # ax.set_xlim([-1.5, 1.5])
                # ax.set_ylim([-1.5, 1.5])

                # Add legend only to the first subplot
                if plot_idx == 0:
                    ax.legend(loc='best', fontsize='small')

    # Remove any unused subplots
    for idx in range(n_plots, len(axes)):
        fig.delaxes(axes[idx])

    # Add overall title
    fig.suptitle("Cognitive Trajectory by Sex")

    # Save the plot
    plt.tight_layout()
    plot_path = os.path.join(output_dir, filename + "_2d.png")
    plt.savefig(plot_path, dpi=dpi)
    plt.close()

    # Add the path to the saved plots
    saved_plots["plots"].append(plot_path)

    # Create 3D plot if we have at least 3 components
    if n_components >= 3:
        fig = plt.figure(figsize=(8 * n_cols, 6 * n_rows))
        ax = fig.add_subplot(111, projection='3d')

        # Plot all points with low alpha for context
        ax.scatter(
            x_scores.iloc[:, 0],
            x_scores.iloc[:, 1],
            x_scores.iloc[:, 2],
            color='lightgray',
            alpha=0.05
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
                x_coord = obs_vect.iloc[row_idx, 0] * factor
                y_coord = obs_vect.iloc[row_idx, 1] * factor
                z_coord = obs_vect.iloc[row_idx, 2] * factor

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
        # ax.set_xlim([-1.5, 1.5])
        # ax.set_ylim([-1.5, 1.5])
        # ax.set_zlim([-1.5, 1.5])

        # Save the plot
        plt.tight_layout()
        plot_path = os.path.join(output_dir, filename + "_3d.png")
        plt.savefig(plot_path, dpi=dpi)
        plt.close()

        # Add the path to the saved plots
        saved_plots["plots"].append(plot_path)

    return saved_plots



def plot_plsr_biplot_3d(
    model,
    X: pd.DataFrame,
    outcome_names: List[str],
    output_dir: str,
    filename: str = "plsr_biplot_3d_components_1_2_3.png",
    exposures_per_plot: int = 25,
    top_n_labels: int = 10,
    vip_scores: Optional[pd.DataFrame] = None,
    dpi: int = 300,
) -> str:
    """
    Create a 3D PLSR biplot using the first three latent variables (LV1, LV2, LV3).

    The plot shows:
    - Sample scores (gray points)
    - Exposure variables (arrows colored by combined VIP over LV1+LV2+LV3)
    - Outcome variables (red arrows)

    VIP handling
    ------------
    - By default, VIP coloring is computed as the combined VIP over components [1, 2, 3]
      using analysis.calculate_vip_x_scores. If that fails, falls back to provided vip_scores
      (combining columns Component_1..Component_3 if present) or uniform coloring.

    Parameters
    ----------
    model : PLSRegression
        Fitted PLSRegression model with x_scores_, x_loadings_, y_loadings_.
    X : pd.DataFrame
        Predictor matrix used to fit the model (for exposure names and ordering).
    outcome_names : List[str]
        Names of outcome variables corresponding to y_loadings_.
    output_dir : str
        Directory where to save the figure.
    filename : str, default="plsr_biplot_3d_components_1_2_3.png"
        Output filename.
    exposures_per_plot : int, default=25
        Maximum number of exposure arrows to draw (top by VIP).
    top_n_labels : int, default=10
        Number of exposure labels to annotate among the drawn set.
    vip_scores : Optional[pd.DataFrame]
        Optional precomputed VIP table with columns 'Component_1', 'Component_2', 'Component_3', ...
    dpi : int, default=300
        Figure DPI.

    Returns
    -------
    str
        Path to saved file, or empty string if fewer than 3 components are available.
    """
    # Validate available components (need at least 3)
    n_comp_scores = getattr(model, "x_scores_", np.empty((0, 0))).shape[1]
    n_comp_xload = getattr(model, "x_loadings_", np.empty((0, 0))).shape[1]
    n_comp_yload = (
        getattr(model, "y_loadings_", np.empty((0, 0))).shape[1]
        if hasattr(model, "y_loadings_")
        else 0
    )
    max_avail = max(0, min(n_comp_scores, n_comp_xload, n_comp_yload if n_comp_yload > 0 else n_comp_scores))
    if max_avail < 3:
        return ""

    exp_names = list(X.columns)

    # Compute combined VIP over components 1,2,3
    def _vip123() -> pd.Series:
        try:
            from excog_trajectory import analysis as _analysis

            comb = _analysis.calculate_vip_x_scores(model, X, components=[1, 2, 3])
            aligned = comb.reindex(exp_names)
            vals = aligned.iloc[:, 0]
        except Exception:
            if vip_scores is not None:
                aligned = vip_scores.reindex(exp_names)
                cols = [c for c in ["Component_1", "Component_2", "Component_3"] if c in aligned.columns]
                if cols:
                    vals = np.sqrt(np.sum(aligned[cols] ** 2, axis=1) / max(len(cols), 1))
                elif "Cumulative" in aligned.columns:
                    vals = aligned["Cumulative"]
                else:
                    vals = pd.Series(1.0, index=exp_names)
            else:
                vals = pd.Series(1.0, index=exp_names)
        # Handle NaNs
        if vals.isna().any():
            fill_val = float(np.nanmin(vals.values)) if np.any(~np.isnan(vals.values)) else 0.0
            vals = vals.fillna(fill_val)
        return vals

    vip_vals = _vip123()

    # Determine which exposures to draw (top by VIP)
    k = min(exposures_per_plot, len(exp_names))
    order = np.argsort(-vip_vals.values)
    draw_indices = list(order[:k])

    # Normalize colors based on drawn exposures
    cmap = plt.cm.Blues
    if draw_indices:
        vmin = float(vip_vals.values[draw_indices].min())
        vmax = float(vip_vals.values[draw_indices].max())
        if np.isclose(vmin, vmax):
            vmin = vmax - 1e-6
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        colors = {idx: cmap(norm(vip_vals.iloc[idx])) for idx in draw_indices}
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
    else:
        colors = {}
        sm = None

    # Slice scores and loadings for the first three components (0-indexed)
    scores = model.x_scores_[:, [0, 1, 2]]
    x_load = model.x_loadings_[:, [0, 1, 2]]
    y_load = model.y_loadings_[:, [0, 1, 2]] if hasattr(model, "y_loadings_") else None

    # Compute scaling so arrows fit within score space
    # Use max score radius in 3D
    max_score = float(np.max(np.sqrt(np.sum(scores**2, axis=1)))) if scores.size > 0 else 1.0
    mats = [x_load]
    if y_load is not None and y_load.size > 0:
        mats.append(y_load)
    max_loading_vec = max(
        1e-12
        + max(
            float(np.max(np.sqrt(np.sum(m**2, axis=1)))) if m.size > 0 else 0.0
            for m in mats
        ),
        1e-6,
    )
    arrow_scale = (max_score * 0.9) / max_loading_vec

    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter sample scores
    ax.scatter(scores[:, 0], scores[:, 1], scores[:, 2], color="lightgray", alpha=0.05, s=9, label="Samples")

    # Draw exposure arrows
    for idx in draw_indices:
        name = exp_names[idx]
        dx, dy, dz = arrow_scale * x_load[idx, 0], arrow_scale * x_load[idx, 1], arrow_scale * x_load[idx, 2]
        ax.quiver(0, 0, 0, dx, dy, dz, color=colors.get(idx, "#1f77b4"), alpha=0.9, arrow_length_ratio=0.08, linewidth=1.0)

    # Label top exposures among drawn
    label_count = min(top_n_labels, len(draw_indices))
    for idx in draw_indices[:label_count]:
        name = exp_names[idx]
        dx, dy, dz = arrow_scale * x_load[idx, 0], arrow_scale * x_load[idx, 1], arrow_scale * x_load[idx, 2]
        ax.text(dx * 1.05, dy * 1.05, dz * 1.05, name, fontsize=9, ha="center", va="center",
                color=colors.get(idx, "#1f77b4"))

    # Draw outcome arrows and labels
    if y_load is not None and len(outcome_names) == y_load.shape[0]:
        for jdx, oname in enumerate(outcome_names):
            dx, dy, dz = arrow_scale * y_load[jdx, 0], arrow_scale * y_load[jdx, 1], arrow_scale * y_load[jdx, 2]
            ax.quiver(0, 0, 0, dx, dy, dz, color="red", alpha=0.9, arrow_length_ratio=0.08, linewidth=1.2)
            ax.text(dx * 1.05, dy * 1.05, dz * 1.05, oname, fontsize=10, ha="center", va="center", color="darkred")

    # Axis labels and title
    ax.set_xlabel("LV1")
    ax.set_ylabel("LV2")
    ax.set_zlabel("LV3")
    ax.set_title("PLSR biplot (LV1 vs LV2 vs LV3)")

    # Set symmetric limits around 0 based on data and arrows
    all_extent = [scores[:, 0].max(), scores[:, 1].max(), scores[:, 2].max(),
                  scores[:, 0].min(), scores[:, 1].min(), scores[:, 2].min()]
    arrow_extents = []
    for mat in [x_load, y_load] if y_load is not None else [x_load]:
        if mat is None:
            continue
        arrow_extents.extend(list(arrow_scale * mat[:, 0]))
        arrow_extents.extend(list(arrow_scale * mat[:, 1]))
        arrow_extents.extend(list(arrow_scale * mat[:, 2]))
    lim = max(1e-6, np.max(np.abs(np.array(all_extent + arrow_extents))))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    # Add colorbar for VIP if available
    if sm is not None:
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("VIP (LV1 + LV2 + LV3)")

    out_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
