"""
Analysis functions for examining relationships between exposures and cognitive decline.

This module provides statistical methods and modeling approaches for analyzing
the associations between environmental exposures and cognitive outcomes in NHANES data.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def run_plsr_cv(
        data: pd.DataFrame,
        exposure_vars: List[str],
        cognitive_vars: List[str],
        covariates: List[str],
        n_components_range: List[int] = None,
        outer_folds: int = 8,
        inner_folds: int = 7,
        scale: bool = True,
        random_state: int = 42,
        n_repetitions: int = 1,
) -> Dict[str, object]:
    """
    Run Partial Least Squares Regression (PLSR) with double cross-validation.

    Uses an outer cross-validation loop to evaluate model performance and an inner
    cross-validation loop to optimize the number of components by maximizing AUROC.
    The process is repeated n_repetitions times, and the best model is selected as
    the most common number of latent variables evaluated in the outer loop.
    A final model is trained on the entire dataset using this optimal number of components.

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
    n_components_range : list of int, default=None
        Range of number of components to try in the inner cross-validation loop.
        If None, will use [1, 2, 3, 4, 5]
    outer_folds : int, default=8
        Number of folds for the outer cross-validation loop
    inner_folds : int, default=7
        Number of folds for the inner cross-validation loop
    scale : bool, default=True
        Whether to standardize the data before running PLSR
    random_state : int, default=42
        Random state for reproducibility
    n_repetitions : int, default=1
        Number of times to repeat the double cross-validation process

    Returns
    -------
    Dict[str, object]
        Dictionary containing the cross-validation results, including:
        - 'best_n_components': List of best number of components for each outer fold across all repetitions
        - 'outer_scores': AUROC scores for each outer fold across all repetitions
        - 'inner_scores': AUROC scores for each inner fold and each number of components
        - 'models': List of fitted PLSR models for each outer fold in the last repetition
        - 'X_vars': List of X variables used
        - 'Y_vars': List of Y variables used
        - 'final_best_n_components': The most common number of components across all outer folds and repetitions
        - 'final_model': The model trained on the entire dataset using the final best number of components
    """
    # TODO: check the inner outer loops and what models are saved

    # Create a copy of the data to avoid modifying the original
    data_plsr = data.copy()

    # Set default range of components if not provided
    if n_components_range is None:
        n_components_range = list(range(1, 6))  # [1, 2, 3, 4, 5]

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

    # Initialize results dictionary
    results = {
        'best_n_components'         : [],
        'outer_scores'              : [],
        'inner_scores'              : {n_comp: [] for n_comp in n_components_range},
        'models'                    : [],
        'X_vars'                    : X_vars,
        'Y_vars'                    : cognitive_vars,
        'repetition_best_components': [],  # Track best components for each repetition
    }

    # Repeat the cross-validation process n_repetitions times
    for rep in range(n_repetitions):
        # Use a different random state for each repetition
        rep_random_state = random_state + rep if random_state is not None else None

        # Initialize outer cross-validation
        outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=rep_random_state)

        # Store best components for this repetition
        rep_best_components = []

        # Outer cross-validation loop
        for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            # Initialize inner cross-validation
            inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=rep_random_state)

            # Dictionary to store inner fold scores for each number of components
            inner_fold_scores = {n_comp: [] for n_comp in n_components_range}

            # Inner cross-validation loop
            for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_train)):
                X_inner_train, X_inner_val = X_train[inner_train_idx], X_train[inner_val_idx]
                Y_inner_train, Y_inner_val = Y_train[inner_train_idx], Y_train[inner_val_idx]

                # Try different numbers of components
                for n_comp in n_components_range:
                    # Fit PLS model
                    plsr = PLSRegression(n_components=n_comp)
                    plsr.fit(X_inner_train, Y_inner_train)

                    # Predict on validation set
                    Y_inner_pred = plsr.predict(X_inner_val)

                    # Calculate AUROC for each cognitive variable and average
                    auroc_scores = []
                    for i in range(Y_inner_val.shape[1]):
                        # Convert predictions to binary for AUROC calculation
                        # Using median as threshold for simplicity
                        y_true_binary = (Y_inner_val[:, i] > np.median(Y_inner_val[:, i])).astype(int)
                        y_pred_score = Y_inner_pred[:, i]

                        try:
                            auroc = roc_auc_score(y_true_binary, y_pred_score)
                            auroc_scores.append(auroc)
                        except ValueError:
                            # Handle case where all true values are the same class
                            auroc_scores.append(0.5)  # Random guess

                    # Average AUROC across cognitive variables
                    mean_auroc = np.mean(auroc_scores)
                    inner_fold_scores[n_comp].append(mean_auroc)

            # Calculate average AUROC for each number of components across inner folds
            mean_inner_scores = {n_comp: np.mean(scores) for n_comp, scores in inner_fold_scores.items()}

            # Find the best number of components
            best_n_comp = max(mean_inner_scores, key=mean_inner_scores.get)
            results['best_n_components'].append(best_n_comp)
            rep_best_components.append(best_n_comp)

            # Store inner fold scores (only for the last repetition to save memory)
            if rep == n_repetitions - 1:
                for n_comp, scores in inner_fold_scores.items():
                    results['inner_scores'][n_comp].extend(scores)

            # Train final model with best number of components on all training data
            best_model = PLSRegression(n_components=best_n_comp)
            best_model.fit(X_train, Y_train)

            # Only store models from the last repetition to save memory
            if rep == n_repetitions - 1:
                results['models'].append(best_model)

            # Evaluate on test set
            Y_test_pred = best_model.predict(X_test)

            # Calculate AUROC for each cognitive variable and average
            outer_auroc_scores = []
            for i in range(Y_test.shape[1]):
                # Convert predictions to binary for AUROC calculation
                y_true_binary = (Y_test[:, i] > np.median(Y_test[:, i])).astype(int)
                y_pred_score = Y_test_pred[:, i]

                try:
                    auroc = roc_auc_score(y_true_binary, y_pred_score)
                    outer_auroc_scores.append(auroc)
                except ValueError:
                    # Handle case where all true values are the same class
                    outer_auroc_scores.append(0.5)  # Random guess

            # Average AUROC across cognitive variables
            mean_outer_auroc = np.mean(outer_auroc_scores)
            results['outer_scores'].append(mean_outer_auroc)

        # Store the best components for this repetition
        results['repetition_best_components'].append(rep_best_components)

    # Find the most common number of components across all outer folds and repetitions
    all_best_components = results['best_n_components']
    from collections import Counter
    component_counts = Counter(all_best_components)
    final_best_n_comp = component_counts.most_common(1)[0][0]
    results['final_best_n_components'] = final_best_n_comp

    # Train a final model on the entire dataset using the most common number of components
    final_model = PLSRegression(n_components=final_best_n_comp)
    final_model.fit(X, Y)
    results['final_model'] = final_model

    # Add additional information to results
    results['n_components_range'] = n_components_range
    results['outer_folds'] = outer_folds
    results['inner_folds'] = inner_folds
    results['scale'] = scale
    results['n_repetitions'] = n_repetitions

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
