"""
Analysis functions for examining relationships between exposures and cognitive decline.

This module provides statistical methods and modeling approaches for analyzing
the associations between environmental exposures and cognitive outcomes in NHANES data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def assess_missing_data(
    data_subset: pd.DataFrame,
    exposure_vars: List[str],
    min_observations: int = 30,
    min_pct_observations: float = 0.1,
) -> Dict[str, object]:
    """
    Assess missing data patterns and calculate reliability metrics for variables.

    Parameters
    ----------
    data_subset : pd.DataFrame
        DataFrame containing the variables to analyze
    exposure_vars : List[str]
        List of exposure variables to include in the assessment
    min_observations : int, default=30
        Minimum number of complete observations required for a reliable correlation
    min_pct_observations : float, default=0.1
        Minimum percentage of total observations required for a reliable correlation

    Returns
    -------
    Dict[str, object]
        Dictionary containing:
        - 'missing_info': DataFrame with missing data counts and percentages
        - 'pairwise_complete_count': DataFrame with pairwise complete observation counts
        - 'pairwise_complete_pct': DataFrame with pairwise complete observation percentages
        - 'pairwise_reliability': DataFrame indicating if pairs have enough observations
        - 'reliability_info': DataFrame with reliability metrics for each variable
    """
    # Calculate the total number of observations
    total_obs = len(data_subset)

    # Calculate the minimum number of observations based on percentage
    min_obs_from_pct = int(total_obs * min_pct_observations)

    # Use the maximum of the two thresholds
    effective_min_obs = max(min_observations, min_obs_from_pct)

    # Calculate missing data statistics
    missing_counts = data_subset.isna().sum()
    missing_percentages = (missing_counts / total_obs) * 100

    # Create a DataFrame with missing data information
    missing_info = pd.DataFrame({
        'variable': missing_counts.index,
        'missing_count': missing_counts.values,
        'missing_percentage': missing_percentages.values
    })

    # Calculate pairwise completeness
    pairwise_complete_count = pd.DataFrame(index=exposure_vars,
                                           columns=exposure_vars)
    pairwise_complete_pct = pd.DataFrame(index=exposure_vars, columns=exposure_vars)
    pairwise_reliability = pd.DataFrame(index=exposure_vars, columns=exposure_vars)

    # Calculate pairwise completeness
    for i, var1 in enumerate(exposure_vars):
        for j, var2 in enumerate(exposure_vars):
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

    # Calculate the number of reliable correlations per variable
    reliable_counts = pd.Series(data = np.count_nonzero(pairwise_reliability, axis=1) - 1,
                                index = exposure_vars)  # Subtract 1 to exclude self-correlation
    reliable_percentages = (reliable_counts / (len(exposure_vars) - 1)) * 100

    # Create a DataFrame with reliability information
    reliability_info = pd.DataFrame({
        'variable': reliable_counts.index,
        'reliable_correlations': reliable_counts.values,
        'reliable_percentage': reliable_percentages.values
    })

    # Store missing data assessment results
    missing_assessment = {
        'missing_info': missing_info,
        'pairwise_complete_count': pairwise_complete_count,
        'pairwise_complete_pct': pairwise_complete_pct,
        'pairwise_reliability': pairwise_reliability,
        'reliability_info': reliability_info,
        'effective_min_obs': effective_min_obs
    }

    return missing_assessment


def run_linear_models(
    data: pd.DataFrame,
    outcome_vars: List[str],
    exposure_vars: List[str],
    covariates: Optional[List[str]] = None,
    survey_design: Optional[object] = None,
) -> Dict[str, Dict[str, object]]:
    """
    Run linear regression models to assess relationships between exposures and cognitive outcomes.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the merged cognitive and exposure data
    outcome_vars : list of str
        List of cognitive outcome variables to model
    exposure_vars : list of str
        List of exposure variables to include as predictors
    covariates : list of str, optional
        List of covariate variables to include in the models
    survey_design : object, optional
        Survey design object for incorporating NHANES complex survey design

    Returns
    -------
    Dict[str, Dict[str, object]]
        Dictionary of model results for each outcome variable
    """
    # Placeholder for actual implementation
    return {}


def run_longitudinal_analysis(
    data: pd.DataFrame,
    outcome_var: str,
    exposure_vars: List[str],
    time_var: str,
    id_var: str,
    covariates: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Run longitudinal analysis to assess exposure effects on cognitive trajectories.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing longitudinal data with multiple time points
    outcome_var : str
        Cognitive outcome variable to model
    exposure_vars : list of str
        List of exposure variables to include as predictors
    time_var : str
        Variable indicating time point
    id_var : str
        Variable indicating subject ID
    covariates : list of str, optional
        List of covariate variables to include in the models

    Returns
    -------
    Dict[str, object]
        Dictionary of longitudinal model results
    """
    # Placeholder for actual implementation
    return {}


def run_mixture_analysis(
    data: pd.DataFrame,
    outcome_vars: List[str],
    exposure_vars: List[str],
    covariates: Optional[List[str]] = None,
    n_components: int = 3,
) -> Dict[str, object]:
    """
    Run mixture modeling to identify patterns of exposures related to cognitive outcomes.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the merged cognitive and exposure data
    outcome_vars : list of str
        List of cognitive outcome variables to model
    exposure_vars : list of str
        List of exposure variables to include in the mixture model
    covariates : list of str, optional
        List of covariate variables to include in the models
    n_components : int, default=3
        Number of mixture components to use

    Returns
    -------
    Dict[str, object]
        Dictionary of mixture model results
    """
    # Placeholder for actual implementation
    return {}


def calculate_exposure_indices(
    data: pd.DataFrame,
    exposure_vars: List[str],
    method: str = "sum",
) -> pd.DataFrame:
    """
    Calculate composite exposure indices from multiple exposure variables.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing exposure variables
    exposure_vars : list of str
        List of exposure variables to include in the index
    method : str, default="sum"
        Method for calculating the index ("sum", "mean", "pca", "weighted")

    Returns
    -------
    pd.DataFrame
        DataFrame with the original data and new exposure index variables
    """
    # Placeholder for actual implementation
    return data.copy()


def run_wgcna(
    data: pd.DataFrame,
    exposure_vars: Optional[List[str]] = None,
    covariates: Optional[List[str]] = None,
    cognitive_vars: Optional[List[str]] = None,
    power: int = 6,
    min_module_size: int = 5,
    cut_height: float = 0.25,
    method: str = "average",
    min_observations: int = 30,
    min_pct_observations: float = 0.1,
    min_reliable_correlations_pct: float = 0.3,
    critical_variables: Optional[List[str]] = None,
    assess_missing: bool = True,
) -> Dict[str, object]:
    """
    Apply Weighted Gene Co-expression Network Analysis (WGCNA) to exposure variables.

    This function implements a robust version of WGCNA for exposure variables that handles
    missing data effectively:
    1. Assess missing data patterns and pairwise completeness
    2. Calculate correlation matrix using pairwise complete observations
    3. Apply quality control measures for correlation reliability
    4. Filter variables based on missing data patterns and reliability
    5. Transform correlation to adjacency using soft thresholding
    6. Perform hierarchical clustering
    7. Cut the dendrogram to identify modules/clusters

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the variables to analyze
    exposure_vars : List[str], optional
        List of exposure variables to include in the analysis. If None, all variables 
        except covariates and cognitive variables will be used.
    covariates : List[str], optional
        List of covariate variables to exclude from the analysis
    cognitive_vars : List[str], optional
        List of cognitive variables to exclude from the analysis
    power : int, default=6
        Soft thresholding power for the adjacency calculation
    min_module_size : int, default=5
        Minimum number of variables required in a module/cluster
    cut_height : float, default=0.25
        Height at which to cut the dendrogram to form clusters
    method : str, default="average"
        Linkage method for hierarchical clustering
    min_observations : int, default=30
        Minimum number of complete observations required for a reliable correlation
    min_pct_observations : float, default=0.1
        Minimum percentage of total observations required for a reliable correlation
    min_reliable_correlations_pct : float, default=0.3
        Minimum percentage of reliable correlations required for a variable to be included
    critical_variables : List[str], optional
        List of variables that should be included regardless of missing data patterns
    assess_missing : bool, default=True
        Whether to perform missing data assessment and include it in the results

    Returns
    -------
    Dict[str, object]
        Dictionary containing:
        - 'clusters': Dictionary mapping cluster IDs to lists of variable names
        - 'adjacency': Adjacency matrix
        - 'dendrogram': Linkage matrix for the hierarchical clustering
        - 'labels': Variable labels with their corresponding cluster assignments
        - 'missing_assessment': Dictionary with missing data assessment results (if assess_missing=True)
        - 'filtered_variables': Dictionary with information about filtered variables
    """
    # If exposure_vars is not provided, use all variables except covariates and cognitive_vars
    if exposure_vars is None:
        # Get all column names
        all_vars = data.columns.tolist()

        # Variables to exclude
        exclude_vars = []
        if covariates is not None:
            exclude_vars.extend(covariates)
        if cognitive_vars is not None:
            exclude_vars.extend(cognitive_vars)

        # Filter out excluded variables
        exposure_vars = [var for var in all_vars if var not in exclude_vars]

    # Initialize critical variables if not provided
    if critical_variables is None:
        critical_variables = []

    # Select only the exposure variables
    data_subset = data[exposure_vars].copy()

    # Initialize missing data assessment results
    missing_assessment = {}

    # Calculate the minimum number of observations based on percentage
    total_obs = len(data_subset)
    min_obs_from_pct = int(total_obs * min_pct_observations)
    effective_min_obs = max(min_observations, min_obs_from_pct)

    if assess_missing:
        # Call the missing data assessment function
        missing_assessment = assess_missing_data(
            data_subset=data_subset,
            exposure_vars=exposure_vars,
            min_observations=min_observations,
            min_pct_observations=min_pct_observations
        )

        # Extract reliability_info for later use
        reliability_info = missing_assessment['reliability_info']
        effective_min_obs = missing_assessment['effective_min_obs']

    # Filter variables based on reliability
    min_reliable_correlations = (len(exposure_vars) - 1) * min_reliable_correlations_pct

    # Identify variables to keep and filter
    variables_to_keep = []
    filtered_variables = []
    filtered_reasons = {}

    for var in exposure_vars:
        # Always keep critical variables
        if var in critical_variables:
            variables_to_keep.append(var)
            continue

        if assess_missing:
            # Get reliability information for this variable
            reliable_count = reliability_info.loc[reliability_info['variable'] == var, 'reliable_correlations'].iloc[0]

            # Check if the variable has enough reliable correlations
            if reliable_count >= min_reliable_correlations:
                variables_to_keep.append(var)
            else:
                filtered_variables.append(var)
                filtered_reasons[var] = f"Insufficient reliable correlations: {reliable_count} < {min_reliable_correlations}"
        else:
            # If not assessing missing data, keep all variables
            variables_to_keep.append(var)

    # Check if we have enough variables after filtering
    if len(variables_to_keep) < 2:
        raise ValueError(f"Not enough variables after filtering: {len(variables_to_keep)} < 2")

    # Create a filtered subset with only the variables to keep
    filtered_subset = data_subset[variables_to_keep].copy()

    # Calculate correlation matrix using pairwise complete observations
    corr_matrix = filtered_subset.corr(method='pearson')

    # Handle any remaining NaN values in the correlation matrix
    # Replace NaN with 0 (no correlation) for the adjacency calculation
    corr_matrix = corr_matrix.fillna(0)

    # Transform correlation to adjacency using soft thresholding (power function)
    # WGCNA uses |correlation|^power
    adjacency = np.abs(corr_matrix) ** power

    # Convert adjacency to distance
    # In WGCNA, distance = 1 - adjacency
    distance = 1 - adjacency

    # Handle any NaN values in the distance matrix
    distance = distance.fillna(1)  # Maximum distance for missing correlations

    # Ensure diagonal elements are zero (distance to self)
    np.fill_diagonal(distance.values, 0)

    # Convert distance matrix to condensed form for linkage function
    condensed_distance = squareform(distance)

    # Perform hierarchical clustering
    Z = linkage(condensed_distance, method=method)

    # Cut the dendrogram to identify clusters
    # Use distance threshold (cut_height) to determine clusters
    clusters = fcluster(Z, cut_height, criterion='distance')

    # Create a dictionary mapping cluster IDs to variable names
    cluster_dict = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(variables_to_keep[i])

    # Filter out clusters that are smaller than min_module_size
    cluster_dict = {k: v for k, v in cluster_dict.items() if len(v) >= min_module_size}

    # Create a DataFrame with variable names and their cluster assignments
    labels = pd.DataFrame({
        'variable': variables_to_keep,
        'cluster': clusters
    })

    # Create a dictionary with information about filtered variables
    filtered_info = {
        'filtered_variables': filtered_variables,
        'filtered_reasons': filtered_reasons,
        'kept_variables': variables_to_keep,
        'min_reliable_correlations': min_reliable_correlations,
        'effective_min_observations': effective_min_obs
    }

    # Return results
    results = {
        'clusters': cluster_dict,
        'adjacency': adjacency,
        'dendrogram': Z,
        'labels': labels,
        'filtered_variables': filtered_info
    }

    # Add missing data assessment if requested
    if assess_missing:
        results['missing_assessment'] = missing_assessment

    return results
