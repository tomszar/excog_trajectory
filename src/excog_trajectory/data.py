"""
Data loading and processing functions for NHANES datasets.

This module provides utilities for loading, cleaning, and preprocessing
NHANES data related to cognitive assessments and environmental exposures.
"""

import os
import urllib.request
import pandas as pd
import zipfile
import shutil
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Set


def load_nhanes_data(
    data_path: Optional[str] = None,
    file_names: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load NHANES data for specified survey cycle(s).

    Parameters
    ----------
    data_path : str, optional
        Path to the directory containing NHANES data files. If None, will use the default path.
    file_names : List[str], optional
        List of specific file names to load. If None, will use the default NHANES data names.

    Returns
    -------
    dict[pd.DataFrame]
        Dictionary of DataFrame containing the loaded NHANES data and var descriptions.
    """
    # Use the default path if none is provided
    if data_path is None:
        data_path = "data/raw/nh_99-06/"

    if file_names is None:
        file_names = ["MainTable.csv", "VarDescription.csv"]

    # Check if the files exist
    for file in file_names:
        if not os.path.exists(os.path.join(data_path, file)):
            raise FileNotFoundError(f"NHANES data file '{file}' not found at {data_path}")

    # Load the main NHANES data file as dict
    print(f"Loading NHANES data from {data_path}...")
    main_data = pd.read_csv(os.path.join(data_path, file_names[0]), index_col="SEQN")
    var_description = pd.read_csv(os.path.join(data_path, file_names[1]))

    # Return a dataframe with the loaded data
    return {"main": main_data, "description": var_description}


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
    grouped by survey year (SDDSRVYR), in a wide format.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to check for missing values. Must contain a 'SDDSRVYR' column.

    Returns
    -------
    pd.DataFrame
        DataFrame in wide format where:
        - Each row represents a variable (column) from the original data
        - Each column represents a survey year, showing the percentage of missing data for that variable in that year
    """
    # Check if SDDSRVYR column exists in the data
    if 'SDDSRVYR' not in data.columns:
        raise ValueError("DataFrame must contain a 'SDDSRVYR' column")

    # Initialize an empty list to store results
    results = []

    # Get unique survey years
    survey_years = data['SDDSRVYR'].unique()

    # For each survey year, calculate the percentage of missing values for each column
    for year in survey_years:
        # Filter data for the current survey year
        year_data = data[data['SDDSRVYR'] == year]

        # Get the total number of rows for this survey year
        total_rows = len(year_data)

        # Calculate the percentage of missing values for each column
        for col in data.columns:
            missing_percentage = (year_data[col].isna().sum() / total_rows) * 100

            # Add the result to our list
            results.append({
                'survey_year': year,
                'column_name': col,
                'percentage_missing': missing_percentage
            })

    # Convert the list of dictionaries to a DataFrame
    long_df = pd.DataFrame(results)

    # Pivot the DataFrame to get it in wide format
    # Each row will be a column from the original data
    # Each column will be a survey year
    wide_df = long_df.pivot(index='column_name', columns='survey_year', values='percentage_missing')

    # Rename the columns to make them more descriptive
    wide_df.columns = [f'year_{year}_missing_pct' for year in wide_df.columns]

    # Reset the index to make 'column_name' a regular column
    wide_df = wide_df.reset_index()

    # Sort by the first year's missing percentage (descending)
    if len(wide_df.columns) > 1:  # Make sure there's at least one year column
        wide_df = wide_df.sort_values(wide_df.columns[1], ascending=False)

    return wide_df


def filter_variables(data: pd.DataFrame, vars_to_filter: List[str], vars_to_keep: Optional[List[str]] = None) -> pd.DataFrame:
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


def filter_exposure_variables(nhanes_data: Dict[str, pd.DataFrame], vars_to_keep: Optional[List[str]] = None) -> pd.DataFrame:
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
    exposure_categories = [# "alcohol use",
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
    output_dir: str = "data/raw",
    id: str = "70319",
    filename: str = "nhanes_data.zip",
    direct_url: str = None
) -> str:
    """
    Download NHANES data from Data Dryad using the API or a direct URL and save it to the output directory.

    Parameters
    ----------
    output_dir : str, optional
        Directory to save the downloaded data, by default "data/raw"
    id : str, optional
        The file ID for the dataset in Data Dryad,
        by default "70319"
    filename : str, optional
        Name to save the downloaded file as, by default "nhanes_data.zip"
    direct_url : str, optional
        Direct URL to download the data from, bypassing the Data Dryad API.
        If provided, this will be used instead of constructing a URL from the DOI.

    Returns
    -------
    str
        Path to the downloaded file
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Full path for the output file
    output_path = os.path.join(output_dir, filename)

    # If a direct URL is provided, use it instead of the DOI-based API
    if direct_url:
        download_url = direct_url
        print(f"Downloading NHANES data from direct URL: {direct_url}...")
    else:
        # Construct the API URL using the file ID
        download_url = f"https://datadryad.org/api/v2/files/70319/download"
        print(f"Downloading NHANES data from Data Dryad (file ID: {id})...")
        print("Note: The default file ID (70319) may be outdated. If download fails, try using a direct URL.")

    # Download the file
    try:
        urllib.request.urlretrieve(download_url, output_path)
        print(f"Download complete. File saved to {output_path}")
    except urllib.error.HTTPError as e:
        if e.code == 403:
            print("Access forbidden. This might be due to API rate limiting or authentication requirements.")
        elif e.code == 404:
            print(f"Dataset not found. Please check the URL or DOI.")
            print("Alternative options:")
            print("1. Search for NHANES datasets at: https://datadryad.org/search?q=nhanes")
            print("2. Download directly from the NHANES website: https://www.cdc.gov/nchs/nhanes/index.htm")
            print("3. Use a direct URL with the direct_url parameter")
        raise

    return output_path


def apply_qc_rules(
    data: pd.DataFrame,
    cognitive_vars: List[str],
    covariates: List[str]
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
    print(f"Total variables removed: {len(removed_vars_rule1) + len(removed_vars_rule2) + len(removed_vars_rule3) + len(removed_vars_rule4)}")
    print(f"Variables remaining: {len(vars_passing_qc) + len(excluded_vars)} out of {len(all_vars)}")

    # Use the filter_variables function to get the final dataset
    # First, order the variables: covariates and cognitive_vars first, then the rest
    ordered_vars = []
    ordered_vars.extend(seqn_var)
    ordered_vars.extend(covariates)
    ordered_vars.extend(cognitive_vars)

    # Use filter_variables to get the final dataset with both the passing QC variables and the excluded variables
    result = filter_variables(data_qc, vars_passing_qc, ordered_vars)

    return result


def extract_nhanes_data(
    zip_path: str,
    output_dir: str = "data/raw"
) -> Tuple[str, List[str]]:
    """
    Extract NHANES data from a zip file and move the contents to the output directory.

    Parameters
    ----------
    zip_path : str
        Path to the downloaded zip file
    output_dir : str, optional
        Directory to save the extracted files, by default "data/raw"

    Returns
    -------
    Tuple[str, List[str]]
        A tuple containing the output directory and a list of extracted file paths
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a temporary directory for extraction
    temp_dir = os.path.join(os.path.dirname(zip_path), "temp_extract")
    os.makedirs(temp_dir, exist_ok=True)

    extracted_files = []

    try:
        print(f"Extracting {zip_path}...")

        # Extract the zip file to the temporary directory
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Get a list of all extracted files
        for root, _, files in os.walk(temp_dir):
            for file in files:
                src_path = os.path.join(root, file)
                # Calculate the relative path to maintain directory structure
                rel_path = os.path.relpath(src_path, temp_dir)
                dest_path = os.path.join(output_dir, rel_path)

                # Create destination directory if it doesn't exist
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                # Move the file to the destination
                shutil.move(src_path, dest_path)
                extracted_files.append(dest_path)
                print(f"Moved {rel_path} to {dest_path}")

        print(f"Extraction complete. {len(extracted_files)} files moved to {output_dir}")

    finally:
        # Clean up the temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")

    return output_dir, extracted_files
