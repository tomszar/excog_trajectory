from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ===========================
# General Utility Functions
# ===========================


def center_matrix(
    df: pd.DataFrame, group_col: str, exclude_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Center a matrix grouped by a categorical variable.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to center.
    group_col : str
        Column to group by (e.g., 'Sex').
    exclude_cols : list of str, optional
        Columns to exclude from centering (e.g., group/status columns).

    Returns
    -------
    centered_df : pd.DataFrame
        Centered DataFrame.
    """
    centered_df = df.copy()
    dat_num = df.drop(columns=exclude_cols or [])
    means = dat_num.groupby(df[group_col]).mean()
    for group, mean in means.iterrows():
        mask = df[group_col] == group
        centered_df.loc[mask, dat_num.columns] = dat_num.loc[mask, :] - mean
    return centered_df


def get_model_matrix(
    X: pd.DataFrame,
    drop_indices: Optional[List[int]] = None,
    add_interactions: bool = True,
) -> np.ndarray:
    """
    Generate a model (design) matrix from categorical factors.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame of factors/covariates.
    drop_indices : list of int, optional
        Indices to drop from the encoded matrix (e.g., for reference levels).
    add_interactions : bool
        Whether to add interaction terms.

    Returns
    -------
    model_mat : np.ndarray
        Model matrix with intercept.
    """
    enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    model_mat = enc.fit_transform(X)
    if drop_indices:
        model_mat = np.delete(model_mat, drop_indices, axis=1)
    if add_interactions and model_mat.shape[1] > 2:
        model_mat = np.c_[
            model_mat,
            model_mat[:, 0] * model_mat[:, 2],
            model_mat[:, 1] * model_mat[:, 2],
        ]
    # Add intercept
    model_mat = np.c_[np.ones(model_mat.shape[0]), model_mat]
    return model_mat


def create_model_matrix(
    model_matrix: pd.DataFrame,
    cognitive_cat: List[str],
    dummy_vars: List[str],
    valid_covariates: List[str],
    add_interactions: bool = True,
) -> pd.DataFrame:
    """
    Create a proper model matrix for trajectory analysis.

    This function:
    1. Adds a column of ones at the beginning (for the intercept)
    2. Drops one variable from each set of dummy variables (captured by the intercept)
    3. Drops the DSST_Average column from the cognitive_cat
    4. Scales the valid_covariates
    5. Optionally adds interaction terms between cognitive_cats and sex dummy vars

    Parameters
    ----------
    model_matrix : pd.DataFrame
        DataFrame containing cognitive categories, dummy variables, and covariates.
    cognitive_cat : List[str]
        List of cognitive category column names.
    dummy_vars : List[str]
        List of dummy variable column names.
    valid_covariates : List[str]
        List of valid covariate column names.
    add_interactions : bool, default=True
        Whether to add interaction terms between cognitive categories and sex dummy variables.

    Returns
    -------
    pd.DataFrame
        Properly formatted model matrix for trajectory analysis.
    """
    # Create a copy of the input data
    df = model_matrix.copy()

    # 1. Drop DSST_Average from cognitive_cat
    if "DSST_Average" in df.columns:
        df = df.drop(columns=["DSST_Average"])
        cognitive_cat = [col for col in cognitive_cat if col != "DSST_Average"]

    # 2. Drop one variable from each set of dummy variables
    # Group dummy variables by their prefix
    dummy_prefixes = set()
    for dummy in dummy_vars:
        prefix = dummy.split("_")[0] + "_"
        dummy_prefixes.add(prefix)

    # For each prefix, drop the first dummy variable
    for prefix in dummy_prefixes:
        prefix_dummies = [col for col in dummy_vars if col.startswith(prefix)]
        if prefix_dummies:
            df = df.drop(columns=[prefix_dummies[0]])
            dummy_vars = [col for col in dummy_vars if col != prefix_dummies[0]]

    # 3. Scale the valid covariates
    if valid_covariates:
        scaler = StandardScaler()
        df[valid_covariates] = scaler.fit_transform(df[valid_covariates])

    # 4. Add interaction terms between cognitive categories and sex dummy variables
    if add_interactions:
        sex_dummies = [col for col in dummy_vars if col.startswith("RIAGENDR_")]
        for cog_cat in cognitive_cat:
            for sex_dummy in sex_dummies:
                interaction_name = f"{cog_cat}_{sex_dummy}"
                df[interaction_name] = df[cog_cat] * df[sex_dummy]

    # 5. Add a column of ones at the beginning (intercept)
    df.insert(0, "Intercept", 1)

    return df


def pair_difference(
    df: pd.DataFrame,
    group_col: str,
    state_col: str,
    state1: str,
    state2: str,
    group1: str,
    group2: str,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[float, float]:
    """
    Estimate vector difference in magnitude and direction between two states, grouped.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    group_col : str
        Column name for grouping variable (e.g., 'Sex').
    state_col : str
        Column name for state variable (e.g., 'Status').
    state1, state2 : str
        Names of the two states to compare.
    group1, group2 : str
        Names of the two groups to compare.
    feature_cols : list of str, optional
        Columns to use for calculation.

    Returns
    -------
    angle : float
        Difference in direction (degrees).
    delta : float
        Difference in magnitude.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in {group_col, state_col}]
    means = {}
    for g in [group1, group2]:
        for s in [state1, state2]:
            mask = (df[group_col] == g) & (df[state_col] == s)
            means[(g, s)] = df.loc[mask, feature_cols].mean()
    vec1 = means[(group1, state1)] - means[(group1, state2)]
    vec2 = means[(group2, state1)] - means[(group2, state2)]
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)
    delta = mag1 - mag2
    cos_angle = np.clip(np.inner(vec1, vec2) / (mag1 * mag2), -1.0, 1.0)
    angle = np.arccos(cos_angle) * 180 / np.pi
    return angle, delta


def estimate_difference(
    Y: Union[pd.DataFrame, np.ndarray],
    model_matrix: Union[pd.DataFrame, np.ndarray],
    LS_means: Union[pd.DataFrame, np.ndarray],
    contrast: List[List[int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate parameters angle, delta, and shape given an outcome matrix,
    model matrix, and contrast to compare. This is a comparison of more than two states.

    Parameters
    ----------
    Y : pd.DataFrame or np.ndarray
        Outcome matrix.
    model_matrix : pd.DataFrame or np.ndarray
        Model matrix with intercept.
    LS_means : pd.DataFrame or np.ndarray
        Least-squares means to estimate.
    contrast : list of lists of int
        Indices indicating the groups to compare.

    Returns
    -------
    deltas : np.ndarray
        Matrix of magnitude differences between groups.
    angles : np.ndarray
        Matrix of direction differences (degrees) between groups.
    shapes : np.ndarray
        Matrix of shape distances between groups.
    """
    n_groups = len(contrast)
    betas = estimate_betas(model_matrix, Y)
    obs_vect = pd.DataFrame(np.matmul(LS_means, betas))
    ys = []
    des = []
    angles = np.zeros((n_groups, n_groups))
    deltas = np.zeros((n_groups, n_groups))
    for i in range(n_groups):
        y = _estimate_orientation(obs_vect, contrast[i])
        d = _estimate_size(obs_vect, contrast[i])
        des.append(d)
        ys.append(y)
    shapes = _estimate_shape(obs_vect, contrast)
    for i in range(n_groups):
        comp = i + 1
        while comp < n_groups:
            delta = np.abs(des[i] - des[comp])
            angle = np.arccos(np.inner(ys[i], ys[comp])) * 180 / np.pi
            deltas[i, comp] = delta
            deltas[comp, i] = delta
            angles[i, comp] = angle
            angles[comp, i] = angle
            comp += 1
    return deltas, angles, shapes


def RRPP(
    Y: Union[pd.DataFrame, np.ndarray],
    model_full: Union[pd.DataFrame, np.ndarray],
    model_reduced: Union[pd.DataFrame, np.ndarray],
    LS_means: Union[pd.DataFrame, np.ndarray],
    contrast: List[List[int]],
    permutations: int = 999,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Residual Randomization in a Permutation Procedure to evaluate linear models.

    Parameters
    ----------
    Y : pd.DataFrame or np.ndarray
        Outcome matrix.
    model_full : pd.DataFrame or np.ndarray
        Model matrix for full model, including intercept.
    model_reduced : pd.DataFrame or np.ndarray
        Model matrix for reduced model, including intercept.
    LS_means : pd.DataFrame or np.ndarray
        Least-squares means to estimate.
    contrast : list of lists of int
        Indices indicating the groups to compare.
    permutations : int
        Number of permutations.

    Returns
    -------
    deltas : list of np.ndarray
        List of magnitude difference matrices across permutations.
    angles : list of np.ndarray
        List of direction difference matrices across permutations.
    shapes : list of np.ndarray
        List of shape distance matrices across permutations.
    """
    Y = pd.DataFrame(Y)
    betas_red = estimate_betas(model_reduced, Y)
    y_hat = np.matmul(model_reduced.to_numpy(), betas_red.to_numpy())
    y_hat = pd.DataFrame(y_hat, index=Y.index, columns=Y.columns)
    y_res = Y - y_hat
    ids = y_res.index
    deltas, angles, shapes = [], [], []
    for _ in range(permutations):
        ids_permuted = np.random.permutation(ids)
        y_res_permuted = y_res.loc[ids_permuted, :]
        y_res_permuted.index = y_res.index
        y_random = y_hat + y_res_permuted
        d, a, s = estimate_difference(y_random, model_full, LS_means, contrast)
        deltas.append(d)
        angles.append(a)
        shapes.append(s)
    return deltas, angles, shapes


def estimate_betas(
    X: Union[pd.DataFrame, np.ndarray], Y: Union[pd.DataFrame, np.ndarray]
) -> np.ndarray:
    """
    Estimate the beta coefficients between an outcome matrix and a model matrix.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Model matrix with intercept.
    Y : pd.DataFrame or np.ndarray
        Outcome matrix.

    Returns
    -------
    betas : np.ndarray
        Beta coefficients.
    """
    left = np.matmul(np.transpose(X), X)
    right = np.matmul(np.transpose(X), Y)
    betas = np.matmul(np.linalg.inv(left), right)
    return betas


def get_observed_vectors(X: pd.DataFrame, Y: pd.DataFrame) -> np.ndarray:
    """
    Get means, or observed vectors, from standard LS vectors.

    Parameters
    ----------
    X : pd.DataFrame
        Matrix of predictors (categorical factors).
    Y : pd.DataFrame
        Matrix of outcomes.

    Returns
    -------
    means : np.ndarray
        Mean values (least squares).
    """
    model_full = get_model_matrix(X)
    betas = estimate_betas(model_full, Y)

    # Convert model_full to DataFrame with column names for _get_ls_vectors
    model_df = pd.DataFrame(model_full)
    # Assign column names based on X's structure
    if isinstance(X, pd.DataFrame) and len(X.columns) > 0:
        # First column is intercept, then the columns from X
        model_df.columns = ["Intercept"] + X.columns.tolist()
    ls_matrix = _get_ls_vectors(model_df)
    means = np.matmul(ls_matrix, betas)
    return means


# ===========================
# Private/Helper Functions
# ===========================


def _estimate_size(obs_vect: pd.DataFrame, levels: List[int]) -> float:
    """
    Estimate the size of a trajectory of two or more levels.

    Parameters
    ----------
    obs_vect : pd.DataFrame
        Matrix of observed mean vectors.
    levels : list of int
        List of indices indicating the levels to consider.

    Returns
    -------
    size : float
        Size of the trajectory.
    """
    if not isinstance(obs_vect, pd.DataFrame):
        obs_vect = pd.DataFrame(obs_vect)
    n_levels = len(levels)
    size = 0
    for i, val in enumerate(levels[:-1]):
        y = obs_vect.iloc[val, :] - obs_vect.iloc[levels[i + 1], :]
        d = np.linalg.norm(y)
        size += d
    return size


def _estimate_orientation(
    obs_vect: pd.DataFrame,
    levels: List[int],
) -> np.ndarray:
    """
    Estimate the orientation of a trajectory of two or more levels.

    Parameters
    ----------
    obs_vect : pd.DataFrame
        Matrix of observed mean vectors.
    levels : list of int
        List of indices indicating the levels to consider.

    Returns
    -------
    orientation : np.ndarray
        Orientation vector of the trajectory.
    """
    if not isinstance(obs_vect, pd.DataFrame):
        obs_vect = pd.DataFrame(obs_vect)
    vect = obs_vect.iloc[levels, :]
    k = vect.shape[1]
    vect -= vect.mean(axis=0)
    _, _, V = np.linalg.svd(np.cov(vect.transpose()))
    orientation = V.transpose()[:k, 0]
    c1 = np.dot(orientation, vect.iloc[0, :])
    sign = c1 / np.abs(c1) if c1 != 0 else 1
    if sign < 0:
        orientation = -orientation
    return orientation


def _estimate_shape(
    vectors: Union[pd.DataFrame, np.ndarray], contrast: List[List[int]]
) -> np.ndarray:
    """
    Align shapes using procrustes superimposition and estimate shape differences.

    Parameters
    ----------
    vectors : pd.DataFrame or np.ndarray
        n x k matrix of vectors to align, n = number of points, k = dimensions.
    contrast : list of lists of int
        Indices indicating the groups to compare.

    Returns
    -------
    shape_distance : np.ndarray
        Matrix with shape distances.
    """
    if isinstance(vectors, pd.DataFrame):
        vect_c = vectors.values.copy()
    else:
        vect_c = vectors.copy()
    n_groups = len(contrast)
    n_levels = len(contrast[0])
    n_dimensions = vect_c.shape[1]
    for levels in contrast:
        means = vect_c[levels, :].mean(axis=0)
        vect_c[levels, :] -= means
    # Scale to centroid size
    for levels in contrast:
        centroid = vect_c[levels, :].mean(axis=0)
        cs = np.sqrt(np.sum((vect_c[levels, :] - centroid) ** 2))
        vect_c[levels, :] /= cs
    # Get baseline Euclidean distance
    Qm1 = euclidean_distances(vect_c.reshape((n_groups, n_dimensions * n_levels)))
    Q = np.tril(Qm1).sum()
    temp1 = vect_c.copy()
    temp2 = vect_c.copy()
    iter_count = 0
    while abs(Q) > 1e-5:
        for i, levels in enumerate(contrast):
            b = [x for idx, x in enumerate(contrast) if idx != i]
            M = (
                np.mean([temp1[lev] for lev in b], axis=0)
                if len(b) > 1
                else temp1[b[0]]
            )
            Mp2 = _OPA(M, temp1[levels])
            temp2[levels] = Mp2
        Qm2 = euclidean_distances(temp2.reshape((n_groups, n_dimensions * n_levels)))
        Q = np.tril(Qm1).sum() - np.tril(Qm2).sum()
        Qm1 = Qm2.copy()
        temp1 = temp2.copy()
        iter_count += 1
    shape_distance = Qm2
    return shape_distance


def _OPA(M1: np.ndarray, M2: np.ndarray) -> np.ndarray:
    """
    Rotate M2 to align with M1 using Orthogonal Procrustes Analysis.

    Parameters
    ----------
    M1 : np.ndarray
        Reference matrix.
    M2 : np.ndarray
        Target matrix to rotate.

    Returns
    -------
    Mp2 : np.ndarray
        Rotated matrix.
    """
    X = M1.T @ M2
    U, S, Vh = np.linalg.svd(X)
    S = np.diag(S)
    D = np.sign(S)
    V = Vh.T
    H = V @ D @ U.T
    Mp2 = M2 @ H
    return Mp2


def _get_ls_vectors(model_matrix: pd.DataFrame) -> np.ndarray:
    """
    Generate a least-squares vectors matrix for male and female with low, average, and high cognition.

    Parameters
    ----------
    model_matrix : pd.DataFrame
        Model matrix containing the predictors.

    Returns
    -------
    ls_matrix : np.ndarray
        LS vector matrix with rows for:
        - Female with low cognition
        - Female with average cognition
        - Female with high cognition
        - Male with low cognition
        - Male with average cognition
        - Male with high cognition
    """
    # Get column names from the model matrix
    cols = model_matrix.columns.tolist()

    # Initialize vectors with zeros
    n_cols = len(cols)
    ls_vectors = np.zeros((6, n_cols))

    # Set intercept for all vectors
    if "Intercept" in cols:
        intercept_idx = cols.index("Intercept")
        ls_vectors[:, intercept_idx] = 1

    # Find indices for cognitive levels
    low_idx = cols.index("DSST_Low") if "DSST_Low" in cols else -1
    high_idx = cols.index("DSST_High") if "DSST_High" in cols else -1

    # Find indices for gender
    male_idx = cols.index("RIAGENDR_1.0") if "RIAGENDR_1.0" in cols else -1
    female_idx = cols.index("RIAGENDR_2.0") if "RIAGENDR_2.0" in cols else -1

    # Set cognitive levels and gender for each vector

    # Female with low cognition (row 0)
    if low_idx >= 0:
        ls_vectors[0, low_idx] = 1
    if female_idx >= 0:
        ls_vectors[0, female_idx] = 1

    # Female with average cognition (row 1)
    # Average is represented by setting both low and high to 0
    if female_idx >= 0:
        ls_vectors[1, female_idx] = 1

    # Female with high cognition (row 2)
    if high_idx >= 0:
        ls_vectors[2, high_idx] = 1
    if female_idx >= 0:
        ls_vectors[2, female_idx] = 1

    # Male with low cognition (row 3)
    if low_idx >= 0:
        ls_vectors[3, low_idx] = 1
    if male_idx >= 0:
        ls_vectors[3, male_idx] = 1

    # Male with average cognition (row 4)
    # Average is represented by setting both low and high to 0
    if male_idx >= 0:
        ls_vectors[4, male_idx] = 1

    # Male with high cognition (row 5)
    if high_idx >= 0:
        ls_vectors[5, high_idx] = 1
    if male_idx >= 0:
        ls_vectors[5, male_idx] = 1

    # Handle interaction terms if they exist
    if low_idx >= 0 and male_idx >= 0:
        interaction_low_male = f"DSST_Low_RIAGENDR_1.0"
        if interaction_low_male in cols:
            interaction_idx = cols.index(interaction_low_male)
            ls_vectors[3, interaction_idx] = 1  # Male with low cognition

    if low_idx >= 0 and female_idx >= 0:
        interaction_low_female = f"DSST_Low_RIAGENDR_2.0"
        if interaction_low_female in cols:
            interaction_idx = cols.index(interaction_low_female)
            ls_vectors[0, interaction_idx] = 1  # Female with low cognition

    if high_idx >= 0 and male_idx >= 0:
        interaction_high_male = f"DSST_High_RIAGENDR_1.0"
        if interaction_high_male in cols:
            interaction_idx = cols.index(interaction_high_male)
            ls_vectors[5, interaction_idx] = 1  # Male with high cognition

    if high_idx >= 0 and female_idx >= 0:
        interaction_high_female = f"DSST_High_RIAGENDR_2.0"
        if interaction_high_female in cols:
            interaction_idx = cols.index(interaction_high_female)
            ls_vectors[2, interaction_idx] = 1  # Female with high cognition

    return ls_vectors


def transform_vectors_to_original(
    vectors: Union[pd.DataFrame, np.ndarray],
    plsr_model: Optional[PLSRegression] = None,
    *,
    x_loadings: Optional[np.ndarray] = None,
    x_mean: Optional[np.ndarray] = None,
    x_std: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Transform a list of vectors (in latent variable space) back to the original X matrix values.

    This function takes vectors in the latent variable space (e.g., LV1, LV2, etc.) and
    transforms them back to the original feature space using either a PLSR model or
    the individual components (x_loadings, x_mean, x_std, feature_names).

    Parameters
    ----------
    vectors : pd.DataFrame or np.ndarray
        Vectors in latent variable space to transform.
    plsr_model : PLSRegression, optional
        The fitted PLSR model containing x_loadings_ and other attributes.
        If provided, the individual components parameters are ignored.
    x_loadings : np.ndarray, optional
        The x_loadings_ matrix from a PLSR model. Required if plsr_model is not provided.
    x_mean : np.ndarray, optional
        The mean values used for scaling X in the PLSR model.
    x_std : np.ndarray, optional
        The standard deviation values used for scaling X in the PLSR model.
    feature_names : List[str], optional
        The feature names from the original X matrix. If not provided, generic names will be used.

    Returns
    -------
    pd.DataFrame
        DataFrame with the original X matrix values, using the column names from the original X matrix.

    Notes
    -----
    Either plsr_model or x_loadings must be provided.
    """
    # Ensure vectors is a numpy array
    if isinstance(vectors, pd.DataFrame):
        vectors_array = vectors.values
    else:
        vectors_array = vectors

    # Get the required components either from the model or from the parameters
    if plsr_model is not None:
        # Use the PLSR model attributes
        loadings = plsr_model.x_loadings_

        # Check if the model has scaling attributes
        has_scaling = hasattr(plsr_model, "_x_mean") and hasattr(plsr_model, "_x_std")
        mean_values = plsr_model._x_mean if has_scaling else None
        std_values = plsr_model._x_std if has_scaling else None

        # Get feature names if available
        if hasattr(plsr_model, "feature_names_in_"):
            column_names = plsr_model.feature_names_in_
        else:
            column_names = None
    else:
        # Use the provided parameters
        if x_loadings is None:
            raise ValueError("Either plsr_model or x_loadings must be provided")

        loadings = x_loadings
        mean_values = x_mean
        std_values = x_std
        column_names = feature_names

    # Transform vectors back to original X space
    # X = T * P^T where T are the scores (vectors) and P are the loadings
    original_x = np.dot(vectors_array, loadings.T)

    # If scaling values are provided, reverse the scaling
    if mean_values is not None and std_values is not None:
        original_x = original_x * std_values + mean_values

    # Create DataFrame with original column names or generic names
    if column_names is None:
        column_names = [f"X{i+1}" for i in range(original_x.shape[1])]

    result_df = pd.DataFrame(original_x, columns=column_names)

    return result_df
