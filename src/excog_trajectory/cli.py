#!/usr/bin/env python3
"""
Command-line interface for the excog_trajectory package.

This module provides CLI commands for analyzing exposomic trajectories
of cognitive decline in NHANES data and for downloading NHANES data.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from excog_trajectory import data, analysis, visualization


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze exposomic trajectories of cognitive decline in NHANES"
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Parser for the 'analyze' command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Run analysis on NHANES data"
    )
    analyze_parser.add_argument(
        "--cycle",
        type=str,
        default="2011-2012",
        help="NHANES survey cycle to analyze (e.g., '2011-2012')",
    )
    analyze_parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/nh_99-06/",
        help="Path to file containing NHANES data files",
    )
    analyze_parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results and figures",
    )
    analyze_parser.add_argument(
        "--output-data",
        type=str,
        default="data/processed",
        help="Directory to save processed data",
    )

    # Parser for the 'download' command
    download_parser = subparsers.add_parser(
        "download", help="Download NHANES data"
    )
    download_parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory to save downloaded data",
    )
    download_parser.add_argument(
        "--id",
        type=str,
        default="70319",
        help="File ID for the dataset in Data Dryad",
    )
    download_parser.add_argument(
        "--direct-url",
        type=str,
        default=None,
        help="Direct URL to download the data from, bypassing the Data Dryad API",
    )
    download_parser.add_argument(
        "--filename",
        type=str,
        default="nhanes_data.zip",
        help="Name to save the downloaded file as",
    )

    args = parser.parse_args()

    # If no command is specified, show help and exit
    if args.command is None:
        parser.print_help()
        exit(1)

    return args


def run_analysis(args):
    """Run the analysis pipeline."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_data, exist_ok=True)

    print(f"Loading NHANES data...")
    nhanes_data = data.load_nhanes_data(data_path=args.data_path)

    # Define variables for analysis
    cognitive_vars = ["CFDRIGHT"]  # Cognitive function right responses
    covariates = ["RIDAGEYR", "female", "male", "black", "mexican", "other_hispanic", "other_eth", "SES_LEVEL", "education"]  # Demographics

    # Keep only relevant columns in the main DataFrame
    print("Filtering NHANES data to keep only relevant columns...")
    # Filter exposure variables
    nhanes_data["main"] = data.filter_exposure_variables(nhanes_data, cognitive_vars + covariates)

    # Remove NaN values from cognitive variables
    print("Removing NaN values from cognitive variables...")
    nhanes_data["main"] = data.remove_nan_from_columns(nhanes_data["main"], cognitive_vars)

    # Remove NaN values from demographic variables
    print("Removing NaN values from demographic variables...")
    nhanes_data["main"] = data.remove_nan_from_columns(nhanes_data["main"], covariates)

    # Apply QC rules to all variables except cognitive and covariate variables
    print("Applying QC rules to variables...")
    nhanes_data["main"] = data.apply_qc_rules(nhanes_data["main"], cognitive_vars, covariates)

    # Save the cleaned data
    nhanes_data["main"].to_csv(os.path.join(args.output_data, "cleaned_nhanes.csv"), index=True)
    print(f"Cleaned data saved to {os.path.join(args.output_data, 'cleaned_nhanes.csv')}")

    # Calculate percentage of missing data for each column
    print("Calculating percentage of missing data...")
    missing_data_df = data.get_percentage_missing(nhanes_data["main"])

    # Save the missing data percentages
    missing_data_df.to_csv(os.path.join(args.output_data, "percentage_missing.csv"), index=False)
    print(f"Percentage of missing data saved to {os.path.join(args.output_data, 'percentage_missing.csv')}")

    print("Creating visualizations...")
    # Plot exposure distributions
    fig1 = visualization.plot_distributions(
        data=nhanes_data["main"],
        vars=cognitive_vars,
        save_path=args.output_dir,
    )
    print(f"Exposure distributions plot saved to {os.path.join(args.output_dir, 'distributions.png')}")

    print("Analysis complete!")


def run_download(args):
    """Download NHANES data and extract it to the output directory."""
    # Download the data
    zip_path = data.download_nhanes_data(
        output_dir=args.output_dir,
        id=args.id,
        filename=args.filename,
        direct_url=args.direct_url
    )

    # Extract the downloaded zip file
    output_dir, extracted_files = data.extract_nhanes_data(
        zip_path=zip_path,
        output_dir=args.output_dir
    )

    print(f"NHANES data successfully downloaded and extracted to {output_dir}")
    print(f"Total files extracted: {len(extracted_files)}")


def main():
    """Main entry point for the CLI."""
    args = parse_args()

    # Execute the appropriate command
    if args.command == "analyze":
        run_analysis(args)
    elif args.command == "download":
        run_download(args)
    else:
        print(f"Unknown command: {args.command}")
        exit(1)


if __name__ == "__main__":
    main()
