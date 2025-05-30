"""
Unit tests for the data module.
"""

import pytest
import pandas as pd
import os
import urllib.request
from unittest.mock import patch, MagicMock
from excog_trajectory import data


def test_load_nhanes_data():
    """Test that load_nhanes_data returns a dictionary of DataFrames."""
    # This is a placeholder test since we don't have actual data
    result = data.load_nhanes_data(cycle="2011-2012")
    assert isinstance(result, dict)


def test_get_cognitive_data():
    """Test that get_cognitive_data returns a DataFrame."""
    # Create a mock NHANES data dictionary
    mock_nhanes_data = {
        "demographics": pd.DataFrame({"SEQN": [1, 2, 3]}),
        "cognitive": pd.DataFrame({
            "SEQN": [1, 2, 3],
            "CFDDS": [10, 20, 30]
        })
    }

    result = data.get_cognitive_data(mock_nhanes_data)
    assert isinstance(result, pd.DataFrame)


def test_get_exposure_data():
    """Test that get_exposure_data returns a DataFrame."""
    # Create a mock NHANES data dictionary
    mock_nhanes_data = {
        "demographics": pd.DataFrame({"SEQN": [1, 2, 3]}),
        "laboratory": pd.DataFrame({
            "SEQN": [1, 2, 3],
            "LBXBPB": [1.0, 2.0, 3.0]
        })
    }

    result = data.get_exposure_data(mock_nhanes_data)
    assert isinstance(result, pd.DataFrame)


def test_merge_cognitive_exposure_data():
    """Test that merge_cognitive_exposure_data correctly merges DataFrames."""
    # Create mock cognitive and exposure DataFrames
    cognitive_data = pd.DataFrame({
        "SEQN": [1, 2, 3],
        "CFDDS": [10, 20, 30]
    })

    exposure_data = pd.DataFrame({
        "SEQN": [1, 2, 3],
        "LBXBPB": [1.0, 2.0, 3.0]
    })

    result = data.merge_cognitive_exposure_data(cognitive_data, exposure_data)
    assert isinstance(result, pd.DataFrame)
    # In a real test, we would check that the merge was done correctly


@patch('urllib.request.urlretrieve')
@patch('os.makedirs')
def test_download_nhanes_data(mock_makedirs, mock_urlretrieve):
    """Test that download_nhanes_data correctly downloads data using file ID."""
    # Setup mock return value
    test_output_path = "test/path/nhanes_data.zip"
    mock_urlretrieve.return_value = (test_output_path, None)

    # Test with default parameters
    result = data.download_nhanes_data()

    # Check that the correct URL was constructed with the file ID
    expected_url = f"https://datadryad.org/api/v2/files/70319/download"
    mock_urlretrieve.assert_called_once_with(expected_url, os.path.join("data/raw", "nhanes_data.zip"))
    assert result == os.path.join("data/raw", "nhanes_data.zip")

    # Reset mock for next test
    mock_urlretrieve.reset_mock()

    # Test with custom file ID
    custom_id = "12345"
    result = data.download_nhanes_data(id=custom_id)

    # Check that the correct URL was constructed with the custom file ID
    expected_url = f"https://datadryad.org/api/v2/files/12345/download"
    mock_urlretrieve.assert_called_once_with(expected_url, os.path.join("data/raw", "nhanes_data.zip"))
    assert result == os.path.join("data/raw", "nhanes_data.zip")

    # Reset mock for next test
    mock_urlretrieve.reset_mock()

    # Test with direct URL
    direct_url = "https://example.com/data.zip"
    result = data.download_nhanes_data(direct_url=direct_url)

    # Check that the direct URL was used
    mock_urlretrieve.assert_called_once_with(direct_url, os.path.join("data/raw", "nhanes_data.zip"))
    assert result == os.path.join("data/raw", "nhanes_data.zip")
