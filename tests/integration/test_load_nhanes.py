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
    # Check if the data file exists
    data_path = 'data/raw/nh_99-06/MainTable.csv'
    if not os.path.exists(data_path):
        pytest.skip(f"Data file not found at {data_path}. Run download and extract first.")
    
    # Load the data
    nhanes_data = load_nhanes_data()
    
    # Check that the data was loaded correctly
    assert isinstance(nhanes_data, dict)
    assert 'main' in nhanes_data
    assert isinstance(nhanes_data['main'], pd.DataFrame)
    assert len(nhanes_data['main']) > 0
    
    # Check that the main data has expected columns
    main_data = nhanes_data['main']
    assert 'SEQN' in main_data.columns  # Subject ID should be present
    
    # Print some information about the data for debugging
    print(f"Data keys: {list(nhanes_data.keys())}")
    print(f"Main data shape: {nhanes_data['main'].shape}")
    print(f"Main data columns: {nhanes_data['main'].columns.tolist()[:5]}...")  # Show first 5 columns


@pytest.mark.integration
def test_load_nhanes_data_custom_path():
    """Test loading NHANES data from a custom path."""
    # This test uses a mock path that doesn't exist, so we mock the pandas.read_csv function
    with pytest.MonkeyPatch.context() as mp:
        # Create a mock DataFrame to return
        mock_df = pd.DataFrame({'SEQN': [1, 2, 3], 'TEST_COL': [10, 20, 30]})
        
        # Mock the pandas.read_csv function
        def mock_read_csv(path, *args, **kwargs):
            assert path == 'custom/path/to/data.csv'
            return mock_df
        
        mp.setattr(pd, 'read_csv', mock_read_csv)
        
        # Load the data with a custom path
        nhanes_data = load_nhanes_data(data_path='custom/path/to/data.csv')
        
        # Check that the data was loaded correctly
        assert isinstance(nhanes_data, dict)
        assert 'main' in nhanes_data
        assert nhanes_data['main'] is mock_df