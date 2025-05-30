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
    # Mock the pandas.read_csv function to avoid actually reading a file
    with patch('pandas.read_csv') as mock_read_csv:
        # Setup mock return value
        mock_df = pd.DataFrame({'column1': [1, 2, 3]})
        mock_read_csv.return_value = mock_df

        # Test with default path
        result = data.load_nhanes_data()

        # Check that the function tried to read the correct file
        mock_read_csv.assert_called_once_with('data/raw/nh_99-06/MainTable.csv')

        # Check that the result is a dictionary with the key 'main' pointing to the mocked DataFrame
        assert isinstance(result, dict)
        assert 'main' in result
        assert result['main'] is mock_df

        # Reset mock for next test
        mock_read_csv.reset_mock()

        # Test with custom path
        custom_path = 'custom/path/to/data.csv'
        result = data.load_nhanes_data(data_path=custom_path)

        # Check that the function tried to read the custom file path
        mock_read_csv.assert_called_once_with(custom_path)

        # Check that the result is a dictionary with the key 'main' pointing to the mocked DataFrame
        assert isinstance(result, dict)
        assert 'main' in result
        assert result['main'] is mock_df


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


def test_remove_nan_from_cfdright():
    """Test that remove_nan_from_cfdright correctly removes rows with NaN values."""
    # Create a DataFrame with some NaN values in CFDRIGHT
    test_data = pd.DataFrame({
        "SEQN": [1, 2, 3, 4, 5],
        "CFDRIGHT": [10, None, 30, float('nan'), 50],
        "OTHER_COL": [1, 2, 3, 4, 5]
    })

    # Call the function
    result = data.remove_nan_from_cfdright(test_data)

    # Check that the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check that rows with NaN in CFDRIGHT were removed
    assert len(result) == 3
    assert all(pd.notna(result["CFDRIGHT"]))
    assert list(result["CFDRIGHT"]) == [10, 30, 50]

    # Check that other columns were preserved
    assert "OTHER_COL" in result.columns
    assert list(result["SEQN"]) == [1, 3, 5]

    # Check that the original DataFrame was not modified
    assert len(test_data) == 5
    assert test_data["CFDRIGHT"].isna().sum() == 2


def test_remove_nan_from_cfdright_no_nan():
    """Test that remove_nan_from_cfdright works correctly when there are no NaN values."""
    # Create a DataFrame with no NaN values in CFDRIGHT
    test_data = pd.DataFrame({
        "SEQN": [1, 2, 3],
        "CFDRIGHT": [10, 20, 30],
        "OTHER_COL": [1, 2, 3]
    })

    # Call the function
    result = data.remove_nan_from_cfdright(test_data)

    # Check that no rows were removed
    assert len(result) == len(test_data)
    assert list(result["CFDRIGHT"]) == [10, 20, 30]


def test_remove_nan_from_cfdright_missing_column():
    """Test that remove_nan_from_cfdright raises an error when CFDRIGHT column is missing."""
    # Create a DataFrame without CFDRIGHT column
    test_data = pd.DataFrame({
        "SEQN": [1, 2, 3],
        "OTHER_COL": [1, 2, 3]
    })

    # Check that the function raises a KeyError
    with pytest.raises(KeyError):
        data.remove_nan_from_cfdright(test_data)


def test_remove_nan_from_columns_single():
    """Test that remove_nan_from_columns correctly removes rows with NaN values in a single column."""
    # Create a DataFrame with some NaN values in a column
    test_data = pd.DataFrame({
        "SEQN": [1, 2, 3, 4, 5],
        "TEST_COL": [10, None, 30, float('nan'), 50],
        "OTHER_COL": [1, 2, 3, 4, 5]
    })

    # Call the function with a single column
    result = data.remove_nan_from_columns(test_data, "TEST_COL")

    # Check that the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check that rows with NaN in the specified column were removed
    assert len(result) == 3
    assert all(pd.notna(result["TEST_COL"]))
    assert list(result["TEST_COL"]) == [10, 30, 50]

    # Check that other columns were preserved
    assert "OTHER_COL" in result.columns
    assert list(result["SEQN"]) == [1, 3, 5]

    # Check that the original DataFrame was not modified
    assert len(test_data) == 5
    assert test_data["TEST_COL"].isna().sum() == 2


def test_remove_nan_from_columns_multiple():
    """Test that remove_nan_from_columns correctly removes rows with NaN values in multiple columns."""
    # Create a DataFrame with some NaN values in multiple columns
    test_data = pd.DataFrame({
        "SEQN": [1, 2, 3, 4, 5, 6],
        "COL1": [10, None, 30, 40, 50, 60],
        "COL2": [1, 2, None, 4, float('nan'), 6],
        "OTHER_COL": [1, 2, 3, 4, 5, 6]
    })

    # Call the function with multiple columns
    result = data.remove_nan_from_columns(test_data, ["COL1", "COL2"])

    # Check that the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check that rows with NaN in any of the specified columns were removed
    assert len(result) == 3
    assert all(pd.notna(result["COL1"]))
    assert all(pd.notna(result["COL2"]))
    assert list(result["SEQN"]) == [1, 4, 6]

    # Check that other columns were preserved
    assert "OTHER_COL" in result.columns

    # Check that the original DataFrame was not modified
    assert len(test_data) == 6
    assert test_data["COL1"].isna().sum() == 1
    assert test_data["COL2"].isna().sum() == 2


def test_remove_nan_from_columns_default():
    """Test that remove_nan_from_columns uses CFDRIGHT as the default column."""
    # Create a DataFrame with some NaN values in CFDRIGHT
    test_data = pd.DataFrame({
        "SEQN": [1, 2, 3, 4, 5],
        "CFDRIGHT": [10, None, 30, float('nan'), 50],
        "OTHER_COL": [1, 2, 3, 4, 5]
    })

    # Call the function without specifying columns (should use CFDRIGHT by default)
    result = data.remove_nan_from_columns(test_data)

    # Check that rows with NaN in CFDRIGHT were removed
    assert len(result) == 3
    assert all(pd.notna(result["CFDRIGHT"]))
    assert list(result["CFDRIGHT"]) == [10, 30, 50]

    # Check that the original DataFrame was not modified
    assert len(test_data) == 5
    assert test_data["CFDRIGHT"].isna().sum() == 2


def test_remove_nan_from_columns_missing_column():
    """Test that remove_nan_from_columns raises an error when a specified column is missing."""
    # Create a DataFrame without the specified column
    test_data = pd.DataFrame({
        "SEQN": [1, 2, 3],
        "OTHER_COL": [1, 2, 3]
    })

    # Check that the function raises a KeyError for a single missing column
    with pytest.raises(KeyError):
        data.remove_nan_from_columns(test_data, "MISSING_COL")

    # Check that the function raises a KeyError for multiple columns where one is missing
    with pytest.raises(KeyError):
        data.remove_nan_from_columns(test_data, ["OTHER_COL", "MISSING_COL"])


def test_get_columns_with_nan():
    """Test that get_columns_with_nan correctly identifies columns with NaN values and their counts."""
    # Create a DataFrame with NaN values in specific columns
    test_data = pd.DataFrame({
        "SEQN": [1, 2, 3, 4, 5],
        "COL_WITH_NAN_1": [10, None, 30, 40, 50],
        "COL_WITHOUT_NAN": [1, 2, 3, 4, 5],
        "COL_WITH_NAN_2": [1, 2, 3, float('nan'), 5]
    })

    # Call the function
    result = data.get_columns_with_nan(test_data)

    # Check that the result is a dictionary
    assert isinstance(result, dict)

    # Check that the result contains exactly the columns with NaN values
    assert len(result) == 2
    assert "COL_WITH_NAN_1" in result
    assert "COL_WITH_NAN_2" in result
    assert "COL_WITHOUT_NAN" not in result
    assert "SEQN" not in result

    # Check that the counts of NaN values are correct
    assert result["COL_WITH_NAN_1"] == 1  # One NaN value
    assert result["COL_WITH_NAN_2"] == 1  # One NaN value

    # Test with a DataFrame that has no NaN values
    test_data_no_nan = pd.DataFrame({
        "COL1": [1, 2, 3],
        "COL2": [4, 5, 6]
    })
    result_no_nan = data.get_columns_with_nan(test_data_no_nan)
    assert isinstance(result_no_nan, dict)
    assert len(result_no_nan) == 0

    # Test with multiple NaN values in a column
    test_data_multiple_nan = pd.DataFrame({
        "COL_WITH_MULTIPLE_NAN": [1, None, 3, None, None]
    })
    result_multiple_nan = data.get_columns_with_nan(test_data_multiple_nan)
    assert isinstance(result_multiple_nan, dict)
    assert len(result_multiple_nan) == 1
    assert "COL_WITH_MULTIPLE_NAN" in result_multiple_nan
    assert result_multiple_nan["COL_WITH_MULTIPLE_NAN"] == 3  # Three NaN values


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
