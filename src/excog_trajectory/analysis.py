"""
Analysis functions for examining relationships between exposures and cognitive decline.

This module provides statistical methods and modeling approaches for analyzing
the associations between environmental exposures and cognitive outcomes in NHANES data.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform


def run_plsr(
        data: pd.DataFrame,
        exposure_vars: List[str],
        cognitive_vars: List[str],
        covariates: List[str],
        n_components: int = 2,
        scale: bool = True,
) -> Dict[str, object]:
    """
    Run Partial Least Squares Regression (PLSR) using exposure variables (including covariates)
    as the X matrix and cognitive variables as the Y matrix.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing exposure, cognitive, and covariate variables
    exposure_vars : list of str
        List of exposure variables to include in the X matrix
    cognitive_vars : list of str
        List of cognitive variables to include in the Y matrix
    covariates : list of str
        List of covariate variables to include in the X matrix
    n_components : int, default=2
        Number of components to use in the PLSR model
    scale : bool, default=True
        Whether to standardize the data before running PLSR

    Returns
    -------
    Dict[str, object]
        Dictionary containing the PLSR model, X and Y loadings, X and Y scores,
        explained variance, and other relevant information
    """
    # Create a copy of the data to avoid modifying the original
    data_plsr = data.copy()

    # Combine exposure variables and covariates for the X matrix
    X_vars = exposure_vars + covariates

    # Check if all variables exist in the data
    missing_x_vars = [var for var in X_vars if var not in data_plsr.columns]
    missing_y_vars = [var for var in cognitive_vars if var not in data_plsr.columns]

    if missing_x_vars:
        raise ValueError(f"The following X variables are missing from the data: {missing_x_vars}")
    if missing_y_vars:
        raise ValueError(f"The following Y variables are missing from the data: {missing_y_vars}")

    # Extract X and Y matrices
    X = data_plsr[X_vars].values
    Y = data_plsr[cognitive_vars].values

    # Standardize the data if requested
    if scale:
        X_scaler = StandardScaler()
        Y_scaler = StandardScaler()
        X = X_scaler.fit_transform(X)
        Y = Y_scaler.fit_transform(Y)

    # Create and fit the PLSR model
    plsr = PLSRegression(n_components=n_components)
    plsr.fit(X, Y)

    # Get the X and Y scores (projections)
    X_scores = plsr.transform(X)
    Y_scores = plsr.y_scores_

    # Calculate explained variance
    x_explained_variance = np.var(X_scores, axis=0) / np.var(X, axis=0).sum()
    y_explained_variance = np.var(Y_scores, axis=0) / np.var(Y, axis=0).sum()

    # Create a dictionary to store the results
    results = {
        "model": plsr,
        "X_loadings": plsr.x_loadings_,
        "Y_loadings": plsr.y_loadings_,
        "X_scores": X_scores,
        "Y_scores": Y_scores,
        "X_explained_variance": x_explained_variance,
        "Y_explained_variance": y_explained_variance,
        "X_vars": X_vars,
        "Y_vars": cognitive_vars,
        "n_components": n_components,
        "X_scaler": X_scaler if scale else None,
        "Y_scaler": Y_scaler if scale else None,
    }

    return results


def run_snf(
        data: pd.DataFrame,
        exposure_categories: Dict[str, List[str]],
        cognitive_vars: List[str],
        covariates: List[str],
        k: int = 20,
        t: int = 20,
        alpha: float = 0.5,
        scale: bool = True,
) -> Dict[str, object]:
    """
    Run Similarity Network Fusion (SNF) using each category of exposure as a block to construct
    the patient similarity network, the cognitive variables as another block, and the covariates
    as another block.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing exposure, cognitive, and covariate variables
    exposure_categories : Dict[str, List[str]]
        Dictionary mapping exposure categories to lists of variable names
    cognitive_vars : list of str
        List of cognitive variables to include as a block
    covariates : list of str
        List of covariate variables to include as a block
    k : int, default=20
        Number of nearest neighbors to consider
    t : int, default=20
        Number of iterations for the fusion process
    alpha : float, default=0.5
        Parameter controlling the importance of local vs. global structure
    scale : bool, default=True
        Whether to standardize the data before running SNF

    Returns
    -------
    Dict[str, object]
        Dictionary containing the fused similarity matrix, individual similarity matrices,
        and other relevant information
    """
    # Create a copy of the data to avoid modifying the original
    data_snf = data.copy()

    # Check if all variables exist in the data
    all_vars = []
    for category, vars_list in exposure_categories.items():
        all_vars.extend(vars_list)
    all_vars.extend(cognitive_vars)
    all_vars.extend(covariates)

    missing_vars = [var for var in all_vars if var not in data_snf.columns]
    if missing_vars:
        raise ValueError(f"The following variables are missing from the data: {missing_vars}")

    # Function to create a similarity matrix from a set of variables
    def create_similarity_matrix(vars_list: List[str]) -> np.ndarray:
        # Extract the data for the variables
        X = data_snf[vars_list].values

        # Standardize the data if requested
        if scale:
            X = StandardScaler().fit_transform(X)

        # Calculate pairwise distances
        dist_matrix = squareform(pdist(X, metric='euclidean'))

        # Convert distances to similarities using a Gaussian kernel
        sigma = np.mean(dist_matrix)
        sim_matrix = np.exp(-dist_matrix**2 / (2 * sigma**2))

        # Set diagonal to 0 (as required by SNF)
        np.fill_diagonal(sim_matrix, 0)

        return sim_matrix

    # Function to find k nearest neighbors
    def find_k_nearest_neighbors(sim_matrix: np.ndarray, k: int) -> np.ndarray:
        n = sim_matrix.shape[0]
        # For each row, keep only the k largest values (excluding self-similarity)
        nn_matrix = np.zeros((n, n))
        for i in range(n):
            # Get indices of k largest values (excluding diagonal)
            idx = np.argsort(sim_matrix[i, :])[-k-1:-1]
            nn_matrix[i, idx] = sim_matrix[i, idx]
        return nn_matrix

    # Function to normalize the similarity matrix
    def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
        row_sums = matrix.sum(axis=1)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        return matrix / row_sums[:, np.newaxis]

    # Create similarity matrices for each block
    similarity_matrices = {}

    # Create similarity matrices for each exposure category
    for category, vars_list in exposure_categories.items():
        if vars_list:  # Only process if the category has variables
            similarity_matrices[category] = create_similarity_matrix(vars_list)

    # Create similarity matrix for cognitive variables
    if cognitive_vars:
        similarity_matrices["cognitive"] = create_similarity_matrix(cognitive_vars)

    # Create similarity matrix for covariates
    if covariates:
        similarity_matrices["covariates"] = create_similarity_matrix(covariates)

    # Prepare matrices for SNF
    normalized_matrices = {}
    k_nearest_matrices = {}

    for name, matrix in similarity_matrices.items():
        # Find k nearest neighbors
        k_nearest = find_k_nearest_neighbors(matrix, min(k, matrix.shape[0]-1))
        # Normalize the matrices
        normalized_matrices[name] = normalize_matrix(k_nearest)
        k_nearest_matrices[name] = k_nearest

    # Perform SNF
    # Initialize fused matrix as the average of all similarity matrices
    matrices_list = list(similarity_matrices.values())
    fused_matrix = sum(matrices_list) / len(matrices_list)

    # Iterative fusion process
    for iteration in range(t):
        for name, matrix in normalized_matrices.items():
            # Update each similarity matrix
            temp = np.zeros_like(fused_matrix)
            for other_name, other_matrix in k_nearest_matrices.items():
                if other_name != name:
                    temp += other_matrix
            temp = temp / (len(normalized_matrices) - 1)

            # Update the fused matrix
            fused_matrix = matrix @ temp @ matrix.T

    # Create a dictionary to store the results
    results = {
        "fused_matrix": fused_matrix,
        "similarity_matrices": similarity_matrices,
        "normalized_matrices": normalized_matrices,
        "k_nearest_matrices": k_nearest_matrices,
        "exposure_categories": exposure_categories,
        "cognitive_vars": cognitive_vars,
        "covariates": covariates,
        "parameters": {
            "k": k,
            "t": t,
            "alpha": alpha,
            "scale": scale,
        },
    }

    return results
