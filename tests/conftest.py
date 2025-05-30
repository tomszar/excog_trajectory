"""
Configuration and fixtures for pytest.
"""

import pytest
import os
import shutil
import pandas as pd


@pytest.fixture
def test_data_dir():
    """Create a temporary directory for test data."""
    test_dir = "test_data_dir"
    os.makedirs(test_dir, exist_ok=True)
    yield test_dir
    # Clean up after the test
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


@pytest.fixture
def test_output_dir():
    """Create a temporary directory for test output."""
    test_dir = "test_output_dir"
    os.makedirs(test_dir, exist_ok=True)
    yield test_dir
    # Clean up after the test
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


@pytest.fixture
def sample_nhanes_data():
    """Create a sample NHANES data dictionary for testing."""
    # Create a sample DataFrame with some test data
    main_data = pd.DataFrame({
        'SEQN': [1, 2, 3, 4, 5],
        'RIDAGEYR': [30, 40, 50, 60, 70],
        'RIAGENDR': [1, 2, 1, 2, 1],  # 1=Male, 2=Female
        'RIDRETH1': [1, 2, 3, 4, 5],  # Race/ethnicity
        'DMDEDUC2': [1, 2, 3, 4, 5],  # Education level
        'INDFMPIR': [0.5, 1.0, 2.0, 3.0, 4.0],  # Poverty income ratio
    })
    
    # Create a sample cognitive data DataFrame
    cognitive_data = pd.DataFrame({
        'SEQN': [1, 2, 3, 4, 5],
        'CFDDS': [10, 20, 30, 40, 50],  # Digit Symbol Substitution Test score
        'CFDRIGHT': [8, 16, 24, 32, 40],  # Number of correct responses
        'CFDTOT': [10, 20, 30, 40, 50],  # Total number of items attempted
    })
    
    # Create a sample laboratory data DataFrame
    laboratory_data = pd.DataFrame({
        'SEQN': [1, 2, 3, 4, 5],
        'LBXBPB': [1.0, 2.0, 3.0, 4.0, 5.0],  # Blood lead level
        'LBXBCD': [0.1, 0.2, 0.3, 0.4, 0.5],  # Blood cadmium level
        'LBXTHG': [0.5, 1.0, 1.5, 2.0, 2.5],  # Blood mercury level
    })
    
    # Return a dictionary of DataFrames
    return {
        'main': main_data,
        'cognitive': cognitive_data,
        'laboratory': laboratory_data
    }


@pytest.fixture
def sample_merged_data():
    """Create a sample merged DataFrame for testing."""
    # Create a sample DataFrame with merged cognitive and exposure data
    merged_data = pd.DataFrame({
        'SEQN': [1, 2, 3, 4, 5],
        'RIDAGEYR': [30, 40, 50, 60, 70],
        'female': [0, 1, 0, 1, 0],
        'male': [1, 0, 1, 0, 1],
        'black': [0, 1, 0, 0, 0],
        'mexican': [0, 0, 1, 0, 0],
        'other_hispanic': [0, 0, 0, 1, 0],
        'other_eth': [1, 0, 0, 0, 1],
        'SES_LEVEL': [1, 2, 3, 4, 5],
        'education': [10, 12, 14, 16, 18],
        'CFDRIGHT': [8, 16, 24, 32, 40],
        'LBXBPB': [1.0, 2.0, 3.0, 4.0, 5.0],
        'LBXBCD': [0.1, 0.2, 0.3, 0.4, 0.5],
    })
    
    return merged_data