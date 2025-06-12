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
    """Test that merge_cognitive_exposure_data correctly merges DataFrames using SEQN as the key."""
    # Create mock cognitive and exposure DataFrames
    cognitive_data = pd.DataFrame({
        "SEQN": [1, 2, 3],
        "CFDDS": [10, 20, 30]
    })

    exposure_data = pd.DataFrame({
        "SEQN": [1, 2, 3],
        "LBXBPB": [1.0, 2.0, 3.0]
    })

    # Patch print to suppress output
    with patch('builtins.print'):
        result = data.merge_cognitive_exposure_data(cognitive_data, exposure_data)

    # Check that the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check that the merge was done correctly using SEQN as the key
    assert len(result) == 3
    assert "SEQN" in result.columns
    assert "CFDDS" in result.columns
    assert "LBXBPB" in result.columns
    assert list(result["SEQN"]) == [1, 2, 3]
    assert list(result["CFDDS"]) == [10, 20, 30]
    assert list(result["LBXBPB"]) == [1.0, 2.0, 3.0]

    # Test with different SEQN values to ensure inner join works correctly
    cognitive_data2 = pd.DataFrame({
        "SEQN": [1, 2, 4],
        "CFDDS": [10, 20, 40]
    })

    exposure_data2 = pd.DataFrame({
        "SEQN": [1, 3, 4],
        "LBXBPB": [1.0, 3.0, 4.0]
    })

    # Patch print to suppress output
    with patch('builtins.print'):
        result2 = data.merge_cognitive_exposure_data(cognitive_data2, exposure_data2)

    # Check that only the common SEQN values are in the result (inner join)
    assert len(result2) == 2
    assert list(result2["SEQN"]) == [1, 4]
    assert list(result2["CFDDS"]) == [10, 40]
    assert list(result2["LBXBPB"]) == [1.0, 4.0]


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


def test_filter_variables():
    """Test that filter_variables correctly filters variables while retaining specified ones."""
    # Create a test DataFrame
    test_data = pd.DataFrame({
        "var1": [1, 2, 3],
        "var2": [4, 5, 6],
        "var3": [7, 8, 9],
        "var4": [10, 11, 12]
    })

    # Test case 1: Filter with no vars_to_keep
    result1 = data.filter_variables(test_data, ["var1", "var3"])
    assert list(result1.columns) == ["var1", "var3"]

    # Test case 2: Filter with vars_to_keep
    result2 = data.filter_variables(test_data, ["var1"], ["var4"])
    # Check that vars_to_keep (var4) appears before vars_to_filter (var1)
    assert list(result2.columns) == ["var4", "var1"]

    # Test case 3: Filter with non-existent variables
    result3 = data.filter_variables(test_data, ["var1", "non_existent"], ["var4", "also_non_existent"])
    # Check that vars_to_keep (var4) appears before vars_to_filter (var1)
    assert list(result3.columns) == ["var4", "var1"]

    # Test case 4: Filter with empty vars_to_filter but non-empty vars_to_keep
    result4 = data.filter_variables(test_data, [], ["var2", "var4"])
    # Check that vars_to_keep appear in the order they were specified
    assert list(result4.columns) == ["var2", "var4"]

    # Test case 5: Filter with empty vars_to_filter and empty vars_to_keep
    result5 = data.filter_variables(test_data, [], [])
    assert result5.empty

    # Test case 6: Filter with non-existent variables only
    result6 = data.filter_variables(test_data, ["non_existent"], ["also_non_existent"])
    assert result6.empty

    # Test case 7: Filter with multiple vars_to_keep and vars_to_filter
    result7 = data.filter_variables(test_data, ["var1", "var3"], ["var4", "var2"])
    # Check that vars_to_keep appear first in the order they were specified,
    # followed by vars_to_filter in the order they were specified
    assert list(result7.columns) == ["var4", "var2", "var1", "var3"]


def test_filter_exposure_variables():
    """Test that filter_exposure_variables correctly filters exposure variables while retaining specified ones."""
    # Create a mock NHANES data dictionary
    main_data = pd.DataFrame({
        "var1": [1, 2, 3],  # Exposure variable
        "var2": [4, 5, 6],  # Exposure variable
        "var3": [7, 8, 9],  # Non-exposure variable
        "var4": [10, 11, 12]  # Non-exposure variable to keep
    })

    description_data = pd.DataFrame({
        "var": ["var1", "var2", "var3", "var4"],
        "category": ["alcohol use", "pesticides", "demographics", "demographics"]
    })

    nhanes_data = {
        "main": main_data,
        "description": description_data
    }

    # Test case 1: Filter with no vars_to_keep
    with patch('builtins.print'):  # Suppress print statements
        result1 = data.filter_exposure_variables(nhanes_data)

    # Should only include exposure variables (var1, var2)
    assert set(result1.columns) == {"var1", "var2"}

    # Test case 2: Filter with vars_to_keep
    with patch('builtins.print'):  # Suppress print statements
        result2 = data.filter_exposure_variables(nhanes_data, ["var4"])

    # Should include exposure variables (var1, var2) and var4
    assert set(result2.columns) == {"var1", "var2", "var4"}

    # Test case 3: Filter with non-existent vars_to_keep
    with patch('builtins.print'):  # Suppress print statements
        result3 = data.filter_exposure_variables(nhanes_data, ["var4", "non_existent"])

    # Should include exposure variables (var1, var2) and var4
    assert set(result3.columns) == {"var1", "var2", "var4"}


def test_get_percentage_missing():
    """Test that get_percentage_missing correctly calculates the percentage of missing data for each column by survey year in wide format."""
    # Create a DataFrame with NaN values in specific columns and survey years
    test_data = pd.DataFrame({
        "SDDSRVYR": [1, 1, 2, 2, 2],  # Two survey years: 1 and 2
        "SEQN": [1, 2, 3, 4, 5],
        "COL_WITH_NAN_1": [10, None, 30, 40, 50],  # 20% missing overall, 50% for year 1, 0% for year 2
        "COL_WITHOUT_NAN": [1, 2, 3, 4, 5],  # 0% missing
        "COL_WITH_NAN_2": [1, 2, 3, float('nan'), 5]  # 20% missing overall, 0% for year 1, 33.33% for year 2
    })

    # Call the function
    result = data.get_percentage_missing(test_data)

    # Check that the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check that the result has the expected columns
    assert "column_name" in result.columns
    assert "year_1_missing_pct" in result.columns
    assert "year_2_missing_pct" in result.columns

    # Check that the result contains one row for each column in the original data
    assert len(result) == 5  # 5 columns in the original data

    # Check percentages for each column and year
    seqn_row = result[result["column_name"] == "SEQN"]
    assert seqn_row["year_1_missing_pct"].values[0] == 0.0
    assert seqn_row["year_2_missing_pct"].values[0] == 0.0

    col1_row = result[result["column_name"] == "COL_WITH_NAN_1"]
    assert col1_row["year_1_missing_pct"].values[0] == 50.0  # 1 out of 2 values is NaN
    assert col1_row["year_2_missing_pct"].values[0] == 0.0  # No NaN in year 2

    col2_row = result[result["column_name"] == "COL_WITHOUT_NAN"]
    assert col2_row["year_1_missing_pct"].values[0] == 0.0
    assert col2_row["year_2_missing_pct"].values[0] == 0.0

    col3_row = result[result["column_name"] == "COL_WITH_NAN_2"]
    assert col3_row["year_1_missing_pct"].values[0] == 0.0  # No NaN in year 1
    assert col3_row["year_2_missing_pct"].values[0] == 33.33333333333333  # 1 out of 3 values is NaN

    # Test with a DataFrame that has no NaN values
    test_data_no_nan = pd.DataFrame({
        "SDDSRVYR": [1, 1, 2],
        "COL1": [1, 2, 3],
        "COL2": [4, 5, 6]
    })
    result_no_nan = data.get_percentage_missing(test_data_no_nan)
    assert isinstance(result_no_nan, pd.DataFrame)
    assert len(result_no_nan) == 3  # 3 columns in the original data

    # Check that all percentages are 0.0
    for col in result_no_nan.columns:
        if col != "column_name":  # Skip the column_name column
            assert all(result_no_nan[col] == 0.0)

    # Test with multiple NaN values in a column
    test_data_multiple_nan = pd.DataFrame({
        "SDDSRVYR": [1, 1, 1, 2, 2],
        "COL_WITH_MULTIPLE_NAN": [1, None, 3, None, None]  # 60% missing overall, 33.33% for year 1, 100% for year 2
    })
    result_multiple_nan = data.get_percentage_missing(test_data_multiple_nan)
    assert isinstance(result_multiple_nan, pd.DataFrame)
    assert len(result_multiple_nan) == 2  # 2 columns in the original data

    # Check percentages for each column and year
    col_row = result_multiple_nan[result_multiple_nan["column_name"] == "COL_WITH_MULTIPLE_NAN"]
    assert col_row["year_1_missing_pct"].values[0] == 33.33333333333333  # 1 out of 3 values is NaN
    assert col_row["year_2_missing_pct"].values[0] == 100.0  # 2 out of 2 values are NaN


def test_apply_qc_rules():
    """Test that apply_qc_rules correctly applies QC rules to the dataset."""
    # Create a test DataFrame with variables that should be removed by each rule
    # and cognitive/covariate variables that should be preserved

    # Create 300 rows to ensure we have enough data for the tests
    n_rows = 300

    # Create a DataFrame with:
    # - cognitive_var: a cognitive variable that should be preserved
    # - covariate_var: a covariate variable that should be preserved
    # - few_non_nan: a variable with less than 200 non-NaN values (Rule 1)
    # - categorical_few: a categorical variable with less than 200 values in a category (Rule 2)
    # - mostly_zeros: a variable with 90% of non-NaN values equal to zero (Rule 3)
    # - missing_in_year: a variable with 100% missing data in one survey year (Rule 4)
    # - keep_var: a variable that should pass all QC rules

    # Create the test data
    import numpy as np
    np.random.seed(42)  # For reproducibility

    # Base DataFrame with IDs and survey years
    test_data = pd.DataFrame({
        "SEQN": range(1, n_rows + 1),
        "SDDSRVYR": np.concatenate([np.ones(100), np.ones(100) * 2, np.ones(100) * 3])  # 3 survey years
    })

    # Cognitive variable (should be preserved)
    test_data["cognitive_var"] = np.random.normal(100, 15, n_rows)

    # Covariate variable (should be preserved)
    test_data["covariate_var"] = np.random.normal(50, 10, n_rows)

    # Variable with less than 200 non-NaN values (should be removed by Rule 1)
    few_non_nan = np.random.normal(0, 1, n_rows)
    few_non_nan[200:] = np.nan  # Make 100 values NaN
    test_data["few_non_nan"] = few_non_nan

    # Categorical variable with less than 200 values in a category (should be removed by Rule 2)
    categorical = np.random.choice([1, 2, 3], n_rows)
    categorical[:150] = 1  # Make category 1 have only 150 values
    test_data["categorical_few"] = categorical

    # Variable with 90% of non-NaN values equal to zero (should be removed by Rule 3)
    mostly_zeros = np.zeros(n_rows)
    mostly_zeros[:30] = np.random.normal(1, 0.1, 30)  # Make 10% non-zero
    test_data["mostly_zeros"] = mostly_zeros

    # Variable with 100% missing data in one survey year (should be removed by Rule 4)
    missing_in_year = np.random.normal(5, 1, n_rows)
    # Make all values in survey year 2 (indices 100-199) NaN
    missing_in_year[100:200] = np.nan
    test_data["missing_in_year"] = missing_in_year

    # Variable that should pass all QC rules
    test_data["keep_var"] = np.random.normal(10, 2, n_rows)

    # Define cognitive and covariate variables
    cognitive_vars = ["cognitive_var"]
    covariates = ["covariate_var"]

    # Call the function
    with patch('builtins.print'):  # Suppress print statements
        result = data.apply_qc_rules(test_data, cognitive_vars, covariates)

    # Check that the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check that cognitive and covariate variables are preserved
    assert "cognitive_var" in result.columns
    assert "covariate_var" in result.columns

    # Check that variables that should be removed are not in the result
    assert "few_non_nan" not in result.columns  # Rule 1
    assert "categorical_few" not in result.columns  # Rule 2
    assert "mostly_zeros" not in result.columns  # Rule 3
    assert "missing_in_year" not in result.columns  # Rule 4

    # Check that variables that should pass all QC rules are in the result
    assert "keep_var" in result.columns

    # Check that the original DataFrame was not modified
    assert "few_non_nan" in test_data.columns
    assert "categorical_few" in test_data.columns
    assert "mostly_zeros" in test_data.columns
    assert "missing_in_year" in test_data.columns

    # Check that SEQN, covariates, and cognitive_vars are first in the order of columns
    assert list(result.columns)[:3] == ["SEQN", "covariate_var", "cognitive_var"]
