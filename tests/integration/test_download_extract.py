"""
Integration tests for downloading and extracting NHANES data.
"""

import pytest
import os
import shutil
from excog_trajectory.data import download_nhanes_data, extract_nhanes_data


@pytest.fixture
def test_output_dir():
    """Create a temporary directory for test output."""
    test_dir = "test_output"
    os.makedirs(test_dir, exist_ok=True)
    yield test_dir
    # Clean up after the test
    shutil.rmtree(test_dir)


@pytest.mark.integration
def test_download_extract_pipeline(test_output_dir):
    """Test the full download and extract pipeline."""
    # Skip this test by default since it requires internet connection
    pytest.skip("Skipping integration test that requires internet connection")
    
    # Download the data
    zip_path = download_nhanes_data(
        output_dir=test_output_dir,
        id="70319",
        filename="nhanes_test_data.zip",
        direct_url="https://datadryad.org/api/v2/files/70319/download"
    )
    
    # Check that the zip file was downloaded
    assert os.path.exists(zip_path)
    assert os.path.getsize(zip_path) > 0
    
    # Extract the data
    output_dir, extracted_files = extract_nhanes_data(
        zip_path=zip_path,
        output_dir=test_output_dir
    )
    
    # Check that files were extracted
    assert len(extracted_files) > 0
    assert os.path.exists(output_dir)
    
    # Check that at least one of the extracted files exists
    assert os.path.exists(os.path.join(output_dir, extracted_files[0]))


@pytest.mark.integration
def test_download_with_direct_url(test_output_dir):
    """Test downloading data with a direct URL."""
    # Skip this test by default since it requires internet connection
    pytest.skip("Skipping integration test that requires internet connection")
    
    # Download the data with a direct URL
    zip_path = download_nhanes_data(
        output_dir=test_output_dir,
        direct_url="https://datadryad.org/api/v2/files/70319/download"
    )
    
    # Check that the zip file was downloaded
    assert os.path.exists(zip_path)
    assert os.path.getsize(zip_path) > 0