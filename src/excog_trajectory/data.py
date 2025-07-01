"""
Data loading and processing functions for NHANES datasets.

This module provides utilities for loading, cleaning, and preprocessing
NHANES data related to cognitive assessments and environmental exposures.
"""

import os
import shutil
import urllib.request
from typing import Dict, List, Optional, Tuple, Union

import dill
import miceforest as mf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_nhanes_data(
        data_path: Optional[str] = None,
        file_names: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load NHANES data.

    Parameters
    ----------
    data_path : str, optional
        Path to the directory containing NHANES data files. If None, will use the default path.
    file_names : List[str], optional
        List of specific file names to load. If None, will use the default NHANES data names.

    Returns
    -------
    dict[pd.DataFrame]
        Dictionary of DataFrame containing the loaded NHANES data.
    """
    # Use the default path if none is provided
    if data_path is None:
        data_path = "data/raw/"

    if file_names is None:
        file_names = ["nhanes_data_1.csv", "nhanes_data_2.csv"]

    # Check if the files exist
    for file in file_names:
        if not os.path.exists(os.path.join(data_path, file)):
            raise FileNotFoundError(f"NHANES data file '{file}' not found at {data_path}")

    # Load the NHANES data files
    print(f"Loading NHANES data from {data_path}...")
    data_dict = {}

    for i, file in enumerate(file_names):
        data_key = f"data_{i+1}"
        data_dict[data_key] = pd.read_csv(os.path.join(data_path, file),
                                          index_col="sample",
                                          na_values=["nan"]).drop("sequence", axis=1)

    # Return a dictionary with the loaded data
    return data_dict


def remove_nan_from_columns(data: pd.DataFrame, columns: Union[str, List[str]] = 'CFDRIGHT') -> pd.DataFrame:
    """
    Remove rows with NaN values in the specified column(s).

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the columns to check for NaN values
    columns : Union[str, List[str]], optional
        Column name(s) to check for NaN values, by default 'CFDRIGHT'
        Can be a single column name (string) or a list of column names

    Returns
    -------
    pd.DataFrame
        DataFrame with rows containing NaN values in specified column(s) removed

    Raises
    ------
    KeyError
        If any of the specified columns are not present in the DataFrame
    """
    # Convert single column to list for consistent handling
    if isinstance(columns, str):
        columns = [columns]

    # Check if all columns exist in the DataFrame
    for column in columns:
        if column not in data.columns:
            raise KeyError(f"Column '{column}' not found in the DataFrame")

    # Drop rows where any of the specified columns have NaN values
    cleaned_data = data.dropna(subset=columns)

    # Print information about removed rows
    num_removed = len(data) - len(cleaned_data)
    if num_removed > 0:
        if len(columns) == 1:
            print(f"Removed {num_removed} rows with NaN values in {columns[0]}")
        else:
            print(f"Removed {num_removed} rows with NaN values in columns: {', '.join(columns)}")

    return cleaned_data


def get_columns_with_nan(data: pd.DataFrame) -> Dict[str, int]:
    """
    Get a dictionary of all columns in the DataFrame that contain at least one NaN value,
    along with the count of NaN values in each column.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to check for NaN values

    Returns
    -------
    Dict[str, int]
        Dictionary where keys are column names that contain at least one NaN value
        and values are the counts of NaN values in each column
    """
    # Check each column for NaN values and count them
    columns_with_nan = {col: int(data[col].isna().sum()) for col in data.columns if data[col].isna().any()}
    return columns_with_nan


def get_percentage_missing(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create a DataFrame that shows the name of each column and the percentage of missing data,
    grouped by cycle or survey year, in a wide format.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to check for missing values. Should contain either a 'SDDSRVYR' or 'Cycle' column.
        If neither is present, will calculate overall missing percentages.

    Returns
    -------
    pd.DataFrame
        DataFrame in wide format where:
        - Each row represents a variable (column) from the original data
        - Each column represents a cycle or survey year, showing the percentage of missing data for that variable
    """
    # Initialize an empty list to store results
    results = []

    # Determine which column to use for grouping
    if 'SDDSRVYR' in data.columns:
        group_col = 'SDDSRVYR'
        group_name = 'survey_year'
    elif 'Cycle' in data.columns:
        group_col = 'Cycle'
        group_name = 'cycle'
    else:
        # If neither column exists, calculate overall missing percentages
        total_rows = len(data)
        for col in data.columns:
            missing_percentage = (data[col].isna().sum() / total_rows) * 100
            results.append({
                'column_name': col,
                'percentage_missing': missing_percentage
            })

        # Create DataFrame and sort by missing percentage
        df = pd.DataFrame(results)
        df = df.sort_values('percentage_missing', ascending=False)
        return df

    # Get unique group values
    group_values = data[group_col].unique()

    # For each group value, calculate the percentage of missing values for each column
    for value in group_values:
        # Filter data for the current group value
        group_data = data[data[group_col] == value]

        # Get the total number of rows for this group
        total_rows = len(group_data)

        # Calculate the percentage of missing values for each column
        for col in data.columns:
            missing_percentage = (group_data[col].isna().sum() / total_rows) * 100

            # Add the result to our list
            results.append({
                group_name: value,
                'column_name': col,
                'percentage_missing': missing_percentage
            })

    # Convert the list of dictionaries to a DataFrame
    long_df = pd.DataFrame(results)

    # Pivot the DataFrame to get it in wide format
    wide_df = long_df.pivot(index='column_name', columns=group_name, values='percentage_missing')

    # Rename the columns to make them more descriptive
    if group_col == 'SDDSRVYR':
        wide_df.columns = [f'year_{value}_missing_pct' for value in wide_df.columns]
    else:
        wide_df.columns = [f'cycle_{value}_missing_pct' for value in wide_df.columns]

    # Reset the index to make 'column_name' a regular column
    wide_df = wide_df.reset_index()

    # Sort by the first group's missing percentage (descending)
    if len(wide_df.columns) > 1:  # Make sure there's at least one group column
        wide_df = wide_df.sort_values(wide_df.columns[1], ascending=False)

    return wide_df


def filter_variables(data: pd.DataFrame, vars_to_filter: List[str],
                     vars_to_keep: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Filter a DataFrame to include only specified variables, while optionally retaining others.
    Variables in vars_to_keep will appear at the beginning of the returned DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to filter
    vars_to_filter : List[str]
        List of variables to filter (include in the result)
    vars_to_keep : Optional[List[str]], optional
        List of variables to always keep, regardless of filtering criteria, by default None

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the filtered variables and any variables specified to keep,
        with vars_to_keep appearing first
    """
    # Initialize lists for variables to include
    keep_vars = []
    filter_vars = []

    # Add variables to keep that exist in the data
    if vars_to_keep:
        keep_vars = [var for var in vars_to_keep if var in data.columns]

    # Add variables to filter that exist in the data and are not already in keep_vars
    filter_vars = [var for var in vars_to_filter if var in data.columns and var not in keep_vars]

    # Combine the lists in the desired order: keep_vars first, then filter_vars
    vars_list = keep_vars + filter_vars

    if not vars_list:
        return pd.DataFrame()

    return data[vars_list].copy()


def filter_exposure_variables(nhanes_data: Dict[str, pd.DataFrame],
                              vars_to_keep: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Filter variables from nhanes_data["main"] that belong to specific exposure categories.

    Uses the nhanes_data["description"] dataframe to retrieve the association between
    a variable and its category (columns var and category respectively).

    Parameters
    ----------
    nhanes_data : Dict[str, pd.DataFrame]
        Dictionary containing NHANES data with keys "main" and "description"
    vars_to_keep : Optional[List[str]], optional
        List of variables to always keep, regardless of exposure categories, by default None

    Returns
    -------
    pd.DataFrame
        DataFrame containing only variables from the specified exposure categories
        and any variables specified to keep

    Raises
    ------
    KeyError
        If "main" or "description" keys are not present in nhanes_data
        If "var" or "category" columns are not present in nhanes_data["description"]
    """
    # Check if required keys are present in nhanes_data
    if "main" not in nhanes_data or "description" not in nhanes_data:
        raise KeyError("nhanes_data must contain 'main' and 'description' keys")

    # Check if required columns are present in nhanes_data["description"]
    if "var" not in nhanes_data["description"].columns or "category" not in nhanes_data["description"].columns:
        raise KeyError("nhanes_data['description'] must contain 'var' and 'category' columns")

    # List of exposure categories to retain
    exposure_categories = [  # "alcohol use",
        # "bacterial infection",
        "cotinine",
        "diakyl",
        "dioxins",
        # "food component recall",
        "furans",
        "heavy metals",
        # "housing",
        "hydrocarbons",
        "nutrients",
        # "occupation",
        "pcbs",
        "perchlorate",
        "pesticides",
        "phenols",
        "phthalates",
        "phytoestrogens",
        "polybrominated ethers",
        "polyflourochemicals",
        # "sexual behavior",
        # "smoking behavior",
        # "smoking family",
        # "social support",
        # "street drug",
        # "sun exposure",
        # "supplement use",
        # "viral infection",
        "volatile compounds"]

    # Filter variables that belong to the specified exposure categories
    exposure_vars = list(set(nhanes_data["description"][
                                 nhanes_data["description"]["category"].isin(exposure_categories)
                             ]["var"]))

    # Use the filter_variables function to filter the data
    filtered_data = filter_variables(nhanes_data["main"], exposure_vars, vars_to_keep)

    if filtered_data.empty:
        print("Warning: No exposure variables found in the main data")
        return pd.DataFrame()

    # Count the number of exposure variables (excluding vars_to_keep)
    exposure_vars_count = len([var for var in filtered_data.columns
                               if vars_to_keep is None or var not in vars_to_keep])

    print(f"Filtered {exposure_vars_count} exposure variables from {len(exposure_categories)} categories")
    if vars_to_keep:
        kept_vars_count = len([var for var in filtered_data.columns if var in vars_to_keep])
        print(f"Kept {kept_vars_count} additional variables as specified")

    return filtered_data


def download_nhanes_data(
        direct_url: Union[str, List[str]],
        output_dir: str = "data/raw",
        filename: str = "nhanes_data.csv"
) -> Union[str, List[str]]:
    """
    Download NHANES data from direct URL(s) and save it to the output directory.

    Parameters
    ----------
    output_dir : str, optional
        Directory to save the downloaded data, by default "data/raw"
    filename : str, optional
        Name to save the downloaded file as, by default "nhanes_data.csv"
        If multiple URLs are provided, this will be used as a base name with indices appended.
    direct_url : Union[str, List[str]]
        Direct URL(s) to download the data from.
        Can be a single URL string or a list of URL strings.

    Returns
    -------
    Union[str, List[str]]
        Path to the downloaded file or list of paths to downloaded files if multiple URLs were provided
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Handle the case where direct_url is a list of URLs
    if isinstance(direct_url, list) and len(direct_url) > 0:
        output_paths = []
        for i, url in enumerate(direct_url):
            # Generate a filename for each URL
            if len(direct_url) > 1:
                # If there are multiple URLs, append an index to the filename
                base_name, ext = os.path.splitext(filename)
                current_filename = f"{base_name}_{i+1}{ext}"
            else:
                current_filename = filename

            # Full path for the output file
            output_path = os.path.join(output_dir, current_filename)

            print(f"Downloading NHANES data from direct URL ({i+1}/{len(direct_url)}): {url}...")

            # Download the file
            try:
                urllib.request.urlretrieve(url, output_path)
                print(f"Download complete. File saved to {output_path}")
                output_paths.append(output_path)
            except urllib.error.HTTPError as e:
                if e.code == 403:
                    print("Access forbidden. This might be due to API rate limiting or authentication requirements.")
                elif e.code == 404:
                    print(f"Dataset not found. Please check the URL.")
                    print("Alternative options:")
                    print("1. Search for NHANES datasets at: https://datadryad.org/search?q=nhanes")
                    print("2. Download directly from the NHANES website: https://www.cdc.gov/nchs/nhanes/index.htm")
                raise

        return output_paths

    # Handle the case where direct_url is a single URL
    # Full path for the output file
    output_path = os.path.join(output_dir, filename)

    # Download the file
    download_url = direct_url
    print(f"Downloading NHANES data from direct URL: {direct_url}...")

    try:
        urllib.request.urlretrieve(download_url, output_path)
        print(f"Download complete. File saved to {output_path}")
    except urllib.error.HTTPError as e:
        if e.code == 403:
            print("Access forbidden. This might be due to API rate limiting or authentication requirements.")
        elif e.code == 404:
            print(f"Dataset not found. Please check the URL.")
            print("Alternative options:")
            print("1. Download directly from the NHANES website: https://www.cdc.gov/nchs/nhanes/index.htm")
        raise

    return output_path


def apply_qc_rules(
        data: pd.DataFrame,
        cognitive_vars: List[str],
        covariates: List[str],
        standardize: bool = False,
        log2_transform: bool = False,
) -> pd.DataFrame:
    """
    Apply quality control rules to the NHANES dataset.

    Rules applied:
    1. Remove variables with less than 200 non-NaN values
    2. Remove categorical variables with less than 200 values in a category
    3. Remove variables with 90% of non-NaN values equal to zero
    4. Remove variables with 100% missing data in at least one survey year

    These rules are applied to all variables except cognitive and covariate variables.
    The returned DataFrame will have SEQN, covariates, and cognitive_vars first in the order of columns.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the NHANES data
    cognitive_vars : List[str]
        List of cognitive variables to exclude from QC rules
    covariates : List[str]
        List of covariate variables to exclude from QC rules
    standardize : bool
        Whether to standardize the data after applying QC rules, by default False
    log2_transform : bool
        Whether to apply log2 transformation to the data after applying QC rules, by default False

    Returns
    -------
    pd.DataFrame
        DataFrame with variables that passed QC rules, with SEQN, covariates, and cognitive_vars first
    """
    # Create a copy of the data to avoid modifying the original
    data_qc = data.copy()

    # Ensure SEQN is included in the variables to keep
    seqn_var = ["SEQN"] if "SEQN" in data.columns else []

    # Identify variables to exclude from QC rules
    excluded_vars = seqn_var + cognitive_vars + covariates

    # Get all variables that are not excluded
    all_vars = set(data.columns)
    vars_to_check = list(all_vars - set(excluded_vars))

    # Variables that pass QC rules
    vars_passing_qc = []

    # Track removed variables for reporting
    removed_vars_rule1 = []
    removed_vars_rule2 = []
    removed_vars_rule3 = []
    removed_vars_rule4 = []

    # Apply Rule 1: Remove variables with less than 200 non-NaN values
    for var in vars_to_check:
        non_nan_count = data[var].notna().sum()
        if non_nan_count >= 200:
            vars_passing_qc.append(var)
        else:
            removed_vars_rule1.append(var)

    # Update vars_to_check to only include variables that passed Rule 1
    vars_to_check = vars_passing_qc.copy()
    vars_passing_qc = []

    # Apply Rule 2: Remove categorical variables with less than 200 values in a category
    for var in vars_to_check:
        # Check if variable is categorical (has fewer than 10 unique values)
        unique_values = data[var].dropna().unique()
        if len(unique_values) < 10:  # Heuristic for identifying categorical variables
            # Check if any category has less than 200 values
            value_counts = data[var].value_counts()
            if (value_counts < 200).any():
                removed_vars_rule2.append(var)
            else:
                vars_passing_qc.append(var)
        else:
            vars_passing_qc.append(var)

    # Update vars_to_check to only include variables that passed Rule 2
    vars_to_check = vars_passing_qc.copy()
    vars_passing_qc = []

    # Apply Rule 3: Remove variables with 90% of non-NaN values equal to zero
    for var in vars_to_check:
        non_nan_values = data[var].dropna()
        if len(non_nan_values) > 0:
            zero_percentage = (non_nan_values == 0).mean()
            if zero_percentage >= 0.9:
                removed_vars_rule3.append(var)
            else:
                vars_passing_qc.append(var)
        else:
            vars_passing_qc.append(var)

    # Update vars_to_check to only include variables that passed Rule 3
    vars_to_check = vars_passing_qc.copy()
    vars_passing_qc = []

    # Apply Rule 4: Remove variables with 100% missing data in at least one survey year
    # Check if SDDSRVYR column exists in the data
    if 'SDDSRVYR' in data.columns:
        # Get percentage of missing data by survey year
        missing_by_year = get_percentage_missing(data)

        # Find variables with 100% missing data in at least one survey year
        for var in vars_to_check:
            # Skip variables that are not in the data (should not happen, but just in case)
            if var not in data.columns:
                continue

            # Get missing percentages for this variable across all survey years
            var_missing = missing_by_year[missing_by_year['column_name'] == var]

            # Check if any survey year has 100% missing data
            # Get all columns except 'column_name' (these are the survey year columns)
            year_columns = [col for col in missing_by_year.columns if col != 'column_name']

            # Check if any survey year has 100% missing data
            if any(var_missing[col].values[0] == 100.0 for col in year_columns):
                removed_vars_rule4.append(var)
            else:
                vars_passing_qc.append(var)
    else:
        # If SDDSRVYR column doesn't exist, skip this rule
        print("Warning: SDDSRVYR column not found, skipping Rule 4")
        vars_passing_qc = vars_to_check.copy()

    # Print summary of removed variables
    print(f"QC Rule 1: Removed {len(removed_vars_rule1)} variables with less than 200 non-NaN values")
    print(f"QC Rule 2: Removed {len(removed_vars_rule2)} categorical variables with less than 200 values in a category")
    print(f"QC Rule 3: Removed {len(removed_vars_rule3)} variables with 90% of non-NaN values equal to zero")
    print(f"QC Rule 4: Removed {len(removed_vars_rule4)} variables with 100% missing data in at least one survey year")
    print(
        f"Total variables removed: {len(removed_vars_rule1) + len(removed_vars_rule2) + len(removed_vars_rule3) + len(removed_vars_rule4)}")
    print(f"Variables remaining: {len(vars_passing_qc) + len(excluded_vars)} out of {len(all_vars)}")

    # Use the filter_variables function to get the final dataset
    # First, order the variables: covariates and cognitive_vars first, then the rest
    ordered_vars = []
    ordered_vars.extend(seqn_var)
    ordered_vars.extend(covariates)
    ordered_vars.extend(cognitive_vars)

    if log2_transform:
        print("Applying log2 transformation to data...")
        # Apply log2 transformation to the data
        eps = 1e-10  # Small value to avoid log2(0)
        data_qc[vars_passing_qc] = np.log2(data_qc[vars_passing_qc] + eps)

    # Normalize the data if specified
    if standardize:
        print("Standardizing data by subtracting mean and dividing by standard deviation...")
        # Normalize the data by subtracting the mean and dividing by the standard deviation
        scaler = StandardScaler().fit(data_qc[vars_passing_qc])
        data_qc[vars_passing_qc] = scaler.transform(data_qc[vars_passing_qc])

    # Use filter_variables to get the final dataset with both the passing QC variables and the excluded variables
    result = filter_variables(data_qc, vars_passing_qc, ordered_vars)

    return result


def extract_nhanes_data(
        csv_path: Union[str, List[str]],
        output_dir: str = "data/raw"
) -> Tuple[str, List[str]]:
    """
    Copy NHANES CSV data files to the output directory.

    Parameters
    ----------
    csv_path : Union[str, List[str]]
        Path to the downloaded CSV file or a list of paths to downloaded CSV files
    output_dir : str, optional
        Directory to save the CSV files, by default "data/raw"

    Returns
    -------
    Tuple[str, List[str]]
        A tuple containing the output directory and a list of copied file paths
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert single csv path to list for consistent handling
    if isinstance(csv_path, str):
        csv_paths = [csv_path]
    else:
        csv_paths = csv_path

    copied_files = []

    for file_path in csv_paths:
        # If the file is already in the output directory, no need to copy
        if os.path.dirname(file_path) == os.path.abspath(output_dir):
            copied_files.append(file_path)
            print(f"File {file_path} is already in the output directory")
            continue

        # Create the destination path
        dest_path = os.path.join(output_dir, os.path.basename(file_path))

        # Copy the file to the destination
        shutil.copy2(file_path, dest_path)
        copied_files.append(dest_path)
        print(f"Copied {file_path} to {dest_path}")

    print(f"All files copied to {output_dir}")
    return output_dir, copied_files


def identify_variable_types(data: pd.DataFrame) -> Dict[str, str]:
    """
    Identify the type of each variable in the DataFrame (continuous or categorical).

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the variables to identify

    Returns
    -------
    Dict[str, str]
        Dictionary mapping variable names to their types ('continuous' or 'categorical')
    """
    var_types = {}

    for col in data.columns:
        # Skip columns with all NaN values
        if data[col].isna().all():
            continue

        # Get non-NaN values
        non_nan_values = data[col].dropna()

        # Check if the variable is numeric
        if pd.api.types.is_numeric_dtype(non_nan_values):
            # Check if the variable is categorical (has fewer than 10 unique values)
            unique_values = non_nan_values.unique()
            if len(unique_values) < 10:
                var_types[col] = 'categorical'
            else:
                var_types[col] = 'continuous'
        else:
            # Non-numeric variables are considered categorical
            var_types[col] = 'categorical'

    return var_types


def impute_exposure_variables(
        data_path: str = "data/processed/cleaned_nhanes.csv",
        output_path: Optional[str] = None,
        n_imputations: int = 5,
        random_state: int = 42,
        n_random_vars: Optional[int] = None,
        n_iterations: int = 3,
        tune_parameters: bool = True,
        save_kernel: bool = False,
        load_kernel: Optional[str] = None,
        diagnostic_plots: bool = False,
) -> mf.ImputationKernel:
    """
    Impute missing values in exposure variables using Multiple Imputation by Chained Equations (MICE)
    with miceforest.

    This function:
    1. Loads the cleaned NHANES dataset
    2. Implements MICE imputation using miceforest
    3. Generates multiple imputed datasets
    4. Saves the imputed datasets with appropriate names

    Parameters
    ----------
    data_path : str, optional
        Path to the cleaned NHANES dataset, by default "data/processed/cleaned_nhanes.csv"
    output_path : Optional[str], optional
        Path to save the imputed datasets, by default None
    n_imputations : int, optional
        Number of imputed datasets to generate, by default 5
    random_state : int, optional
        Random state for reproducibility, by default 42
    n_random_vars : Optional[int], optional
        Number of random variables to select for imputation. If None, all variables are used, by default None
    n_iterations : int, optional
        Number of iterations for the imputation procedure, by default 3
    tune_parameters : bool
        Whether to tune the imputation parameters using gradient boosting decision trees (GBDT), by default True.
    save_kernel: bool
        Whether to save the ImputationKernel object after imputation, by default False.
    load_kernel: Optional[str]
        Path to load an existing ImputationKernel object. If provided, this will skip the imputation step.
    diagnostic_plots: bool
        Whether to generate diagnostic plots during the imputation process, by default False.

    Returns
    -------
    ImputationKernel
    """
    if output_path is None:
        output_path = "data/processed/"

    if load_kernel:
        # Load the existing ImputationKernel object
        print(f"Loading ImputationKernel from {load_kernel}...")
        with open(load_kernel, "rb") as f:
            kernel = dill.load(f)
        print("ImputationKernel loaded successfully.")
    elif load_kernel is None:
        # Load the cleaned NHANES dataset
        print(f"Loading cleaned NHANES data from {data_path}...")
        data = pd.read_csv(data_path, index_col=0).reset_index()

        # Store SEQN separately if it exists in the data
        seqn_data = None
        if 'SEQN' in data.columns:
            seqn_data = data['SEQN'].copy()
            print("SEQN column found and will be preserved but not used in imputation")

        demographic_vars = ["RIDAGEYR", "female", "male", "black", "mexican", "other_hispanic",
                            "other_eth", "SES_LEVEL", "education", "SDDSRVYR", "CFDRIGHT"]
        # Exclude SEQN from exposure variables
        exposure_vars = [col for col in data.columns if col not in demographic_vars and col != 'SEQN']

        print(f"Identified {len(exposure_vars)} exposure variables")

        # Randomly select a subset of variables if n_random_vars is provided
        if n_random_vars is not None and n_random_vars < len(exposure_vars):
            # Set random seed for reproducibility
            np.random.seed(random_state)
            # Randomly select n_random_vars variables
            exposure_vars = np.random.choice(exposure_vars, size=n_random_vars, replace=False).tolist()
            print(f"Randomly selected {len(exposure_vars)} variables for imputation")

        # Create a copy of the data for imputation, excluding SEQN
        to_impute = data.loc[:, demographic_vars + exposure_vars].copy()

        # Create a dataset for miceforest
        print("Creating miceforest kernel...")

        # Create the kernel
        kernel = mf.ImputationKernel(
            data=to_impute,
            save_all_iterations_data=True,
            random_state=random_state,
            num_datasets=n_imputations,
        )

        if tune_parameters:
            print(f"Running parameter tunning...")
            optimal_parameters = kernel.tune_parameters(use_gbdt=True)
            print(f"Running imputation with {n_iterations} iterations...")
            kernel.mice(n_iterations, variable_parameters=optimal_parameters)
        elif tune_parameters is False:
            print(f"Running imputation with {n_iterations} iterations...")
            kernel.mice(n_iterations)
        else:
            raise ValueError("tune_parameters must be True or False")

        # Create output directory if it doesn't exist
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Save the kernel if specified
        if save_kernel:
            kernel_path = output_path + "imputation_kernel.pkl" if output_path else "imputation_kernel.pkl"
            print(f"Saving ImputationKernel to {kernel_path}...")
            with open(kernel_path, "wb") as f:
                dill.dump(kernel, f)
    else:
        raise ValueError("Either load_kernel must be provided or the imputation should be performed from scratch.")

    # Generate and save the imputed datasets
    print(f"Generating {n_imputations} imputed datasets...")

    if n_imputations != kernel.num_datasets:
        print(
            f"Warning: n_imputations ({n_imputations}) does not match the number of datasets in the kernel ({kernel.num_datasets})."
            f"Using the number of datasets in the kernel ({kernel.num_datasets}) instead.")
        n_imputations = kernel.num_datasets

    for i in range(n_imputations):
        # Get the imputed dataset
        dataset_num = i + 1
        imputed_dataset = kernel.complete_data(dataset=i)

        # Add SEQN back to the imputed dataset if it was stored
        if seqn_data is not None:
            imputed_dataset['SEQN'] = seqn_data.values
            # Reorder columns to put SEQN first
            cols = ['SEQN'] + [col for col in imputed_dataset.columns if col != 'SEQN']
            imputed_dataset = imputed_dataset[cols]

        # Save the dataset with appropriate name
        filename = "imputed_nhanes_dat" + str(dataset_num) + ".csv"
        print(f"Saving imputed dataset {filename} to {output_path}...")
        imputed_dataset.to_csv(output_path + filename, index=False)

    print("Imputation complete!")

    if diagnostic_plots:
        print("Generating diagnostic plots...")
        distributions = kernel.plot_imputed_distributions()
        distributions.save("results/distributions_mice.png", dpi=300, width=20, height=20)
        if kernel.save_all_iterations_data:
            importance = kernel.plot_feature_importance(kernel.num_datasets - 1)
            importance.save("results/feature_importance_mice.png", dpi=300, width=60, height=60, limitsize=False)

    return kernel
