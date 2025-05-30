"""
Data loading and processing functions for NHANES datasets.

This module provides utilities for loading, cleaning, and preprocessing
NHANES data related to cognitive assessments and environmental exposures.
"""

import os
import urllib.request
import pandas as pd
from typing import Dict, List, Optional, Union


def load_nhanes_data(
    cycle: Union[str, List[str]],
    data_path: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load NHANES data for specified survey cycle(s).

    Parameters
    ----------
    cycle : str or list of str
        NHANES survey cycle(s) to load (e.g., '2011-2012', ['2011-2012', '2013-2014'])
    data_path : str, optional
        Path to the directory containing NHANES data files. If None, will use default path.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary of DataFrames containing the loaded NHANES data, with keys corresponding
        to the different data components (demographics, questionnaire, laboratory, etc.)
    """
    # Placeholder for actual implementation
    return {}


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
