"""
Integration tests for downloading NHANES data.
"""

import pytest
import os
import shutil
from excog_trajectory.data import download_nhanes_data


@pytest.fixture
def test_output_dir():
    """Create a temporary directory for test output."""
    test_dir = "test_output"
    os.makedirs(test_dir, exist_ok=True)
    yield test_dir
    # Clean up after the test
    shutil.rmtree(test_dir)


@pytest.mark.integration
def test_download_pipeline(test_output_dir):
    """Test the download pipeline."""
    # Skip this test by default since it requires internet connection
    pytest.skip("Skipping integration test that requires internet connection")

    # Download the data
    csv_path = download_nhanes_data(
        output_dir=test_output_dir,
        filename="nhanes_test_data.csv",
        direct_url="https://osf.io/download/9aupq/"
    )

    # Check that the CSV file was downloaded
    assert os.path.exists(csv_path)
    assert os.path.getsize(csv_path) > 0


@pytest.mark.integration
def test_download_with_direct_url(test_output_dir):
    """Test downloading data with a direct URL."""
    # Skip this test by default since it requires internet connection
    pytest.skip("Skipping integration test that requires internet connection")

    # Download the data with a direct URL
    csv_path = download_nhanes_data(
        output_dir=test_output_dir,
        direct_url="https://osf.io/download/9vewm/"
    )

    # Check that the CSV file was downloaded
    assert os.path.exists(csv_path)
    assert os.path.getsize(csv_path) > 0


@pytest.mark.integration
def test_download_multiple_files(test_output_dir):
    """Test downloading multiple files."""
    # Skip this test by default since it requires internet connection
    pytest.skip("Skipping integration test that requires internet connection")

    # Download multiple files
    direct_urls = [
        "https://osf.io/download/9aupq/",
        "https://osf.io/download/9vewm/"
    ]

    csv_paths = download_nhanes_data(
        output_dir=test_output_dir,
        filename="nhanes_test_data.csv",
        direct_url=direct_urls
    )

    # Check that the CSV files were downloaded
    assert isinstance(csv_paths, list)
    assert len(csv_paths) == 2
    for csv_path in csv_paths:
        assert os.path.exists(csv_path)
        assert os.path.getsize(csv_path) > 0
