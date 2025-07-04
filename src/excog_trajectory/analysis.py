"""
Analysis functions for examining relationships between exposures and cognitive decline.

This module provides statistical methods and modeling approaches for analyzing
the associations between environmental exposures and cognitive outcomes in NHANES data.
"""

import multiprocessing
from itertools import repeat
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler


def pls_double_cv(
        x: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
        cv1_splits: int = 7,
        cv2_splits: int = 8,
        n_repeats: int = 30,
        max_components: int = 50,
        random_state: int = 1203,
) -> dict[str, Union[PLSRegression, pd.DataFrame]]:
    """
    Estimate a double cross validation on a partial least squares
    regression.

    Parameters
    ----------
    x: pd.DataFrame
        The predictor variables.
    y: Union[pd.DataFrame, pd.Series]
        The outcome variable.
    cv1_splits: int
        Number of folds in the CV1 loop. Default: 7.
    cv2_splits: int
        Number of folds in the CV2 loop. Default: 8.
    n_repeats: int
        Number of repeats to the cv2 procedure. Default: 30.
    max_components: int
        Maximum number of LV to test. Default: 50.
    random_state: int
        For reproducibility. Default: 1203.

    Returns
    -------
    models_table: dict[str,
                       Union[PLSRegression,
                             pd.DataFrame]
        Dictionary with the table of the best models, including repetition,
        number of latent variables, and R2 score. Also includes the model for
        prediction.
    """
    cv2 = RepeatedKFold(
        n_splits=cv2_splits, n_repeats=n_repeats, random_state=random_state
    )
    cv1 = KFold(n_splits=cv1_splits)
    cv2_table = pd.DataFrame(np.zeros((cv2_splits, 2)))
    cv1_table = pd.DataFrame(np.zeros((cv1_splits, 2)))
    for_table = {
        "rep"     : list(range(1, n_repeats + 1)),
        "LV"      : list(range(1, n_repeats + 1)),
        "R2_score": [0.1] * n_repeats,
    }
    model_table = pd.DataFrame(for_table)
    row_cv2 = 0
    row_model_table = 0
    cv2_models = []
    best_models = []
    for rest, test in cv2.split(x, y):
        # Outer CV2 loop split into test and rest
        x_rest = x.iloc[rest, :]
        x_test = x.iloc[test, :]
        y_rest = y.iloc[rest]
        y_test = y.iloc[test]
        row_cv1 = 0
        for train, validation in cv1.split(x_rest, y_rest):
            # Inner CV validates optimal number of LVs
            x_train = x_rest.iloc[train, :]
            y_train = y_rest.iloc[train, :]
            x_val = x_rest.iloc[validation, :]
            y_val = y_rest.iloc[validation, :]
            ns = list(range(1, max_components))
            with multiprocessing.Pool(processes=None) as pool:
                r2_scores = pool.starmap(
                    _plsda_r2,
                    zip(
                        ns,
                        repeat(x_train),
                        repeat(y_train),
                        repeat(x_val),
                        repeat(y_val),
                    ),
                )
            nlv = r2_scores.index(max(r2_scores)) + 1
            cv1_table.iloc[row_cv1, 0] = nlv
            cv1_table.iloc[row_cv1, 1] = max(r2_scores)
            row_cv1 += 1
        # Get optimal n of components
        n_components = int(cv1_table.iloc[cv1_table[1].idxmax(), 0])
        model_score = _plsda_r2(
            n_components, x_rest, y_rest, x_test, y_test, return_full=True
        )
        cv2_table.iloc[row_cv2, 0] = n_components
        cv2_table.iloc[row_cv2, 1] = model_score["score"]
        cv2_models.append(model_score["model"])
        row_cv2 += 1
        if row_cv2 == cv2_splits:
            best_cv2_lv = int(cv2_table.iloc[cv2_table[1].idxmax(), 0])
            r2_val = cv2_table.iloc[cv2_table[1].idxmax(), 1]
            model_table.iloc[row_model_table, 1] = best_cv2_lv
            model_table.iloc[row_model_table, 2] = r2_val
            best_models.append(cv2_models[cv2_table[1].idxmax()])
            row_model_table += 1
            cv2_table = pd.DataFrame(np.zeros((cv2_splits, 2)))
            cv2_models = []
            row_cv2 = 0
    models_table = {"models": best_models, "table": model_table}
    return models_table


def _plsda_r2(
        n_components: int,
        x_train: pd.DataFrame,
        y_train: Union[pd.Series, pd.DataFrame],
        x_test: pd.DataFrame,
        y_test: Union[pd.Series, pd.DataFrame],
        return_full: bool = False,
) -> Union[float, dict[str, Union[PLSRegression, float]]]:
    """
    Estimate a partial least squares regression and return the coefficient of
    determination and the model, or just the coefficient of determination.

    Parameters
    ----------
    n_components: int
        Number of components to use.
    x_train: pd.DataFrame
        The predictors to use for training.
    y_train: Union[pd.Series, pd.DataFrame]
        The outcome to use for training.
    x_test: pd.DataFrame,
        The predictors to use for testing.
    y_test: Union[pd.Series, pd.DataFrame]
        The outcomes to use for testing.
    return_full: bool
        Whether to return the model and R2 score.
        If False, returns only the R2 score. Default: False.

    Returns
    -------
    auroc: Union[float,
                 dict[str,
                      Union[PLSRegression,
                            float]]]
        Return the R2 score, and optionally the regression model.
    """
    pls = PLSRegression(
        n_components=n_components, scale=True, max_iter=1000).fit(
        X=x_train, y=y_train
    )
    score = pls.score(x_test, y_test)
    if return_full:
        r2_score = {"model": pls, "score": score}
    else:
        r2_score = score
    return r2_score


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
