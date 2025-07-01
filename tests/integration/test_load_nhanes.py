"""
Integration tests for loading NHANES data.
"""

import pytest
import os
import pandas as pd
from excog_trajectory.data import load_nhanes_data


@pytest.mark.integration
def test_load_nhanes_data_real_file():
    """Test loading NHANES data from a real file."""
    # Check if the data files exist
    data_path1 = 'data/raw/nhanes_data_1.csv'
    data_path2 = 'data/raw/nhanes_data_2.csv'
    if not os.path.exists(data_path1) or not os.path.exists(data_path2):
        pytest.skip(f"Data files not found at {data_path1} or {data_path2}. Run download and extract first.")

    # Load the data
    nhanes_data = load_nhanes_data()

    # Check that the data was loaded correctly
    assert isinstance(nhanes_data, dict)
    assert 'data_1' in nhanes_data
    assert 'data_2' in nhanes_data
    assert isinstance(nhanes_data['data_1'], pd.DataFrame)
    assert isinstance(nhanes_data['data_2'], pd.DataFrame)
    assert len(nhanes_data['data_1']) > 0
    assert len(nhanes_data['data_2']) > 0

    # Check that the data has expected columns
    data_1 = nhanes_data['data_1']
    data_2 = nhanes_data['data_2']
    assert 'SEQN' in data_1.index.name  # Subject ID should be the index
    assert 'SEQN' in data_2.index.name  # Subject ID should be the index

    # Print some information about the data for debugging
    print(f"Data keys: {list(nhanes_data.keys())}")
    print(f"Data 1 shape: {nhanes_data['data_1'].shape}")
    print(f"Data 2 shape: {nhanes_data['data_2'].shape}")
    print(f"Data 1 columns: {nhanes_data['data_1'].columns.tolist()[:5]}...")  # Show first 5 columns
    print(f"Data 2 columns: {nhanes_data['data_2'].columns.tolist()[:5]}...")  # Show first 5 columns


@pytest.mark.integration
def test_load_nhanes_data_custom_path():
    """Test loading NHANES data from a custom path."""
    # This test uses a mock path that doesn't exist, so we mock the pandas.read_csv function
    with pytest.MonkeyPatch.context() as mp:
        # Create a mock DataFrame to return
        mock_df = pd.DataFrame({'TEST_COL': [10, 20, 30]})
        mock_df.index.name = 'SEQN'

        # Mock the pandas.read_csv function
        def mock_read_csv(path, *args, **kwargs):
            assert path in ['custom/path/to/custom_file_1.csv', 'custom/path/to/custom_file_2.csv']
            assert kwargs.get('index_col') == 'SEQN'
            return mock_df

        mp.setattr(pd, 'read_csv', mock_read_csv)

        # Load the data with a custom path and custom file names
        custom_path = 'custom/path/to/'
        custom_files = ['custom_file_1.csv', 'custom_file_2.csv']
        nhanes_data = load_nhanes_data(data_path=custom_path, file_names=custom_files)

        # Check that the data was loaded correctly
        assert isinstance(nhanes_data, dict)
        assert 'data_1' in nhanes_data
        assert 'data_2' in nhanes_data
        assert nhanes_data['data_1'] is mock_df
        assert nhanes_data['data_2'] is mock_df
