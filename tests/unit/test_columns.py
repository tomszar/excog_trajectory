"""
Unit tests for the cols module.
"""

import pandas as pd
import pytest

from excog_trajectory import columns


def test_validate_columns():
    """Test the validate_columns function."""
    # Create a test DataFrame
    data = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    })
    
    # Test with all valid cols
    valid_cols = columns.validate_columns(data, ['A', 'B'], raise_error=True)
    assert valid_cols == ['A', 'B']
    
    # Test with some invalid cols, but don't raise error
    valid_cols = columns.validate_columns(data, ['A', 'D'], raise_error=False)
    assert valid_cols == ['A']
    
    # Test with invalid cols and raise error
    with pytest.raises(ValueError):
        columns.validate_columns(data, ['D', 'E'], raise_error=True)


def test_get_dummy_prefixes():
    """Test the get_dummy_prefixes function."""
    # Test with default categorical cols
    prefixes = columns.get_dummy_prefixes()
    assert prefixes == ['Cycle_', 'RIAGENDR_', 'RIDRETH1_']
    
    # Test with custom categorical cols
    prefixes = columns.get_dummy_prefixes(['A', 'B'])
    assert prefixes == ['A_', 'B_']


def test_get_exposure_vars():
    """Test the get_exposure_vars function."""
    # Create a test DataFrame with cognitive, covariate, and exposure variables
    data = pd.DataFrame({
        'CFDRIGHT': [1, 2, 3],
        'CFDDS': [4, 5, 6],
        'Cycle': [1, 2, 3],
        'RIDAGEYR': [30, 40, 50],
        'RIAGENDR': [1, 2, 1],
        'Cycle_1': [1, 0, 0],
        'Cycle_2': [0, 1, 0],
        'Exposure1': [0.1, 0.2, 0.3],
        'Exposure2': [0.4, 0.5, 0.6]
    })
    
    # Test with default parameters
    exposure_vars = columns.get_exposure_vars(data)
    assert set(exposure_vars) == {'Exposure1', 'Exposure2'}
    
    # Test with custom cognitive and covariate variables
    exposure_vars = columns.get_exposure_vars(
        data,
        cognitive_vars=['CFDRIGHT'],
        covariates=['Cycle', 'RIDAGEYR']
    )
    assert set(exposure_vars) == {'CFDDS', 'RIAGENDR', 'Exposure1', 'Exposure2'}


def test_categorize_exposure_vars():
    """Test the categorize_exposure_vars function."""
    # Create a test DataFrame with exposure variables
    data = pd.DataFrame({
        'LBXPB': [0.1, 0.2, 0.3],  # Lead (heavy metal)
        'LBXCOT': [0.4, 0.5, 0.6],  # Cotinine
        'LBXBPA': [0.7, 0.8, 0.9],  # Bisphenol A (phenol)
        'OtherVar': [1, 2, 3]  # Not matching any pattern
    })
    
    # Test with default patterns
    categories = columns.categorize_exposure_vars(data)
    assert 'heavy metals' in categories
    assert 'cotinine' in categories
    assert 'phenols' in categories
    assert 'LBXPB' in categories['heavy metals']
    assert 'LBXCOT' in categories['cotinine']
    assert 'LBXBPA' in categories['phenols']
    
    # Test with custom patterns
    custom_patterns = {
        'metals': ['PB'],
        'other': ['Other']
    }
    categories = columns.categorize_exposure_vars(data, patterns=custom_patterns)
    assert 'metals' in categories
    assert 'other' in categories
    assert 'LBXPB' in categories['metals']
    assert 'OtherVar' in categories['other']