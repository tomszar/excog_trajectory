"""
Unit tests for the CLI module.
"""

import pytest
import os
import pandas as pd
from unittest.mock import patch, MagicMock
from excog_trajectory import cli


def test_parse_args_clean():
    """Test that parse_args correctly parses clean command arguments."""
    with patch('sys.argv', ['excog_trajectory', 'clean', '--cycle', '2013-2014', '--data-path', 'custom/path', '--output-dir', 'custom/output']):
        args = cli.parse_args()

        assert args.command == 'clean'
        assert args.cycle == '2013-2014'
        assert args.data_path == 'custom/path'
        assert args.output_dir == 'custom/output'


def test_parse_args_download():
    """Test that parse_args correctly parses download command arguments."""
    with patch('sys.argv', ['excog_trajectory', 'download', '--output-dir', 'custom/download', '--filename', 'custom.csv']):
        args = cli.parse_args()

        assert args.command == 'download'
        assert args.output_dir == 'custom/download'
        assert args.filename == 'custom.csv'
        assert args.direct_url == ["https://osf.io/download/9aupq/", "https://osf.io/download/9vewm/"]


def test_parse_args_download_direct_url():
    """Test that parse_args correctly parses download command with direct URL."""
    with patch('sys.argv', ['excog_trajectory', 'download', '--direct-url', 'https://example.com/data.csv']):
        args = cli.parse_args()

        assert args.command == 'download'
        assert args.direct_url == ['https://example.com/data.csv']


def test_parse_args_no_command():
    """Test that parse_args exits when no command is provided."""
    with patch('sys.argv', ['excog_trajectory']):
        with patch('sys.exit') as mock_exit:
            cli.parse_args()
            mock_exit.assert_called_once_with(1)


@patch('excog_trajectory.data.load_nhanes_data')
@patch('excog_trajectory.data.get_cognitive_data')
@patch('excog_trajectory.data.get_exposure_data')
@patch('excog_trajectory.data.merge_cognitive_exposure_data')
@patch('excog_trajectory.data.remove_nan_from_columns')
@patch('excog_trajectory.analysis.run_linear_models')
@patch('excog_trajectory.visualization.plot_exposure_distributions')
@patch('excog_trajectory.visualization.plot_exposure_outcome_relationships')
@patch('excog_trajectory.visualization.plot_model_coefficients')
@patch('os.makedirs')
@patch('pandas.DataFrame.to_csv')
@patch('matplotlib.figure.Figure.savefig')
def test_clean_data(mock_savefig, mock_to_csv, mock_makedirs, mock_plot_coef, 
                     mock_plot_relationships, mock_plot_dist, mock_run_models, 
                     mock_remove_nan, mock_merge, mock_get_exposure, 
                     mock_get_cognitive, mock_load_nhanes):
    """Test that clean_data executes the data cleaning pipeline without errors."""
    # Setup mock return values
    mock_load_nhanes.return_value = {'main': pd.DataFrame()}
    mock_get_cognitive.return_value = pd.DataFrame()
    mock_get_exposure.return_value = pd.DataFrame()
    mock_merge.return_value = pd.DataFrame({
        'CFDRIGHT': [1, 2, 3],
        'RIDAGEYR': [30, 40, 50],
        'female': [1, 0, 1],
        'male': [0, 1, 0],
        'black': [0, 1, 0],
        'mexican': [0, 0, 1],
        'other_hispanic': [0, 0, 0],
        'other_eth': [1, 0, 0],
        'SES_LEVEL': [1, 2, 3],
        'education': [12, 16, 14],
        'LBXBPB': [1.0, 2.0, 3.0],
        'LBXBCD': [0.1, 0.2, 0.3]
    })
    mock_remove_nan.return_value = pd.DataFrame()
    mock_run_models.return_value = {}
    mock_plot_dist.return_value = MagicMock()
    mock_plot_relationships.return_value = MagicMock()
    mock_plot_coef.return_value = MagicMock()

    # Create args object
    args = MagicMock()
    args.cycle = '2011-2012'
    args.data_path = 'test/data'
    args.output_dir = 'test/output'

    # Call the function
    cli.clean_data(args)

    # Check that all the expected functions were called
    mock_makedirs.assert_called_once_with('test/output', exist_ok=True)
    mock_load_nhanes.assert_called_once_with(data_path='test/data')
    mock_get_cognitive.assert_called_once()
    mock_get_exposure.assert_called_once()
    mock_merge.assert_called_once()
    assert mock_remove_nan.call_count == 2  # Called for cognitive and demographic variables
    mock_to_csv.assert_called_once()
    mock_run_models.assert_called_once()
    mock_plot_dist.assert_called_once()
    assert mock_plot_relationships.call_count == 1  # Called for each cognitive variable
    mock_plot_coef.assert_called_once()
    assert mock_savefig.call_count == 3  # Called for each plot


@patch('excog_trajectory.data.download_nhanes_data')
def test_run_download(mock_download):
    """Test that run_download executes the download pipeline without errors."""
    # Setup mock return values
    mock_download.return_value = 'test/path/to/csv'

    # Create args object
    args = MagicMock()
    args.output_dir = 'test/output'
    args.filename = 'test.csv'
    args.direct_url = 'https://example.com/data.csv'

    # Call the function
    cli.run_download(args)

    # Check that all the expected functions were called
    mock_download.assert_called_once_with(
        output_dir='test/output',
        filename='test.csv',
        direct_url='https://example.com/data.csv'
    )


@patch('excog_trajectory.cli.parse_args')
@patch('excog_trajectory.cli.clean_data')
@patch('excog_trajectory.cli.run_download')
def test_main_clean(mock_run_download, mock_clean_data, mock_parse_args):
    """Test that main correctly calls clean_data for the clean command."""
    # Setup mock return value
    args = MagicMock()
    args.command = 'clean'
    mock_parse_args.return_value = args

    # Call the function
    cli.main()

    # Check that clean_data was called with the args
    mock_clean_data.assert_called_once_with(args)
    mock_run_download.assert_not_called()


@patch('excog_trajectory.cli.parse_args')
@patch('excog_trajectory.cli.clean_data')
@patch('excog_trajectory.cli.run_download')
def test_main_download(mock_run_download, mock_clean_data, mock_parse_args):
    """Test that main correctly calls run_download for the download command."""
    # Setup mock return value
    args = MagicMock()
    args.command = 'download'
    mock_parse_args.return_value = args

    # Call the function
    cli.main()

    # Check that run_download was called with the args
    mock_run_download.assert_called_once_with(args)
    mock_clean_data.assert_not_called()


@patch('excog_trajectory.cli.parse_args')
@patch('excog_trajectory.cli.clean_data')
@patch('excog_trajectory.cli.run_download')
@patch('sys.exit')
def test_main_unknown_command(mock_exit, mock_run_download, mock_clean_data, mock_parse_args):
    """Test that main exits for an unknown command."""
    # Setup mock return value
    args = MagicMock()
    args.command = 'unknown'
    mock_parse_args.return_value = args

    # Call the function
    cli.main()

    # Check that sys.exit was called with code 1
    mock_exit.assert_called_once_with(1)
    mock_clean_data.assert_not_called()
    mock_run_download.assert_not_called()
