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
from typing import Dict, List, Optional, Union, Tuple


def load_nhanes_data(
    data_path: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load NHANES data for specified survey cycle(s).

    Parameters
    ----------
    data_path : str, optional
        Path to the directory containing NHANES data files. If None, will use default path.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the loaded NHANES data.
    """
    # Use the default path if none is provided
    if data_path is None:
        data_path = 'data/raw/nh_99-06/MainTable.csv'

    # Check if the file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"NHANES data file not found at {data_path}")

    # Load the main NHANES data file
    print(f"Loading NHANES data from {data_path}...")
    main_data = pd.read_csv(data_path)

    # Return a dictionary with the loaded data
    # The key 'main' is used for the main dataset
    return main_data


def get_cognitive_data(
    nhanes_data: Dict[str, pd.DataFrame],
    measures: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Extract cognitive assessment data from NHANES datasets.

    Parameters
    ----------
    nhanes_data : Dict[str, pd.DataFrame]
        Dictionary of NHANES DataFrames as returned by load_nhanes_data()
    measures : list of str, optional
        List of specific cognitive measures to extract. If None, extracts all available measures.

    Returns
    -------
    pd.DataFrame
        DataFrame containing cognitive assessment data
    """
    # Placeholder for actual implementation
    return pd.DataFrame()


def get_exposure_data(
    nhanes_data: Dict[str, pd.DataFrame],
    exposure_categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Extract environmental exposure data from NHANES datasets.

    Parameters
    ----------
    nhanes_data : Dict[str, pd.DataFrame]
        Dictionary of NHANES DataFrames as returned by load_nhanes_data()
    exposure_categories : list of str, optional
        List of specific exposure categories to extract (e.g., ['metals', 'pesticides']).
        If None, extracts all available exposure data.

    Returns
    -------
    pd.DataFrame
        DataFrame containing environmental exposure data
    """
    # Placeholder for actual implementation
    return pd.DataFrame()


def merge_cognitive_exposure_data(
    cognitive_data: pd.DataFrame,
    exposure_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge cognitive assessment and exposure data into a single dataset.

    Parameters
    ----------
    cognitive_data : pd.DataFrame
        DataFrame containing cognitive assessment data
    exposure_data : pd.DataFrame
        DataFrame containing environmental exposure data

    Returns
    -------
    pd.DataFrame
        Merged DataFrame containing both cognitive and exposure data
    """
    # Placeholder for actual implementation
    return pd.DataFrame()


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


def remove_nan_from_cfdright(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with NaN values in the CFDRIGHT column.

    This function is maintained for backward compatibility.
    It is recommended to use remove_nan_from_columns() instead.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the CFDRIGHT column

    Returns
    -------
    pd.DataFrame
        DataFrame with rows containing NaN values in CFDRIGHT removed

    Raises
    ------
    KeyError
        If the CFDRIGHT column is not present in the DataFrame
    """
    return remove_nan_from_columns(data, 'CFDRIGHT')


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
