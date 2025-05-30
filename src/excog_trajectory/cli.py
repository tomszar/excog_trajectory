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
        default="data/raw",
        help="Path to directory containing NHANES data files",
    )
    analyze_parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results and figures",
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

    print(f"Loading NHANES data for cycle {args.cycle}...")
    nhanes_data = data.load_nhanes_data(cycle=args.cycle, data_path=args.data_path)

    print("Extracting cognitive assessment data...")
    cognitive_data = data.get_cognitive_data(nhanes_data)

    print("Extracting environmental exposure data...")
    exposure_data = data.get_exposure_data(nhanes_data)

    print("Merging cognitive and exposure data...")
    merged_data = data.merge_cognitive_exposure_data(cognitive_data, exposure_data)

    # Save the merged data
    merged_data.to_csv(os.path.join(args.output_dir, "merged_data.csv"), index=False)
    print(f"Merged data saved to {os.path.join(args.output_dir, 'merged_data.csv')}")

    # Define variables for analysis
    outcome_vars = ["CFDDS", "CFDST"]  # Digit Symbol Substitution Test scores
    exposure_vars = ["LBXBPB", "LBXBCD"]  # Blood lead and cadmium levels
    covariates = ["RIDAGEYR", "RIAGENDR", "RIDRETH1", "DMDEDUC2"]  # Demographics

    print("Running linear models...")
    model_results = analysis.run_linear_models(
        data=merged_data,
        outcome_vars=outcome_vars,
        exposure_vars=exposure_vars,
        covariates=covariates
    )

    print("Creating visualizations...")
    # Plot exposure distributions
    fig1 = visualization.plot_exposure_distributions(
        data=merged_data,
        exposure_vars=exposure_vars
    )
    fig1.savefig(os.path.join(args.output_dir, "exposure_distributions.png"))
    print(f"Exposure distributions plot saved to {os.path.join(args.output_dir, 'exposure_distributions.png')}")

    # Plot exposure-outcome relationships
    for outcome_var in outcome_vars:
        fig2 = visualization.plot_exposure_outcome_relationships(
            data=merged_data,
            outcome_var=outcome_var,
            exposure_vars=exposure_vars
        )
        fig2.savefig(os.path.join(args.output_dir, f"{outcome_var}_relationships.png"))
        print(f"Exposure-outcome relationships plot saved to {os.path.join(args.output_dir, f'{outcome_var}_relationships.png')}")

    # Plot model coefficients
    fig3 = visualization.plot_model_coefficients(
        model_results=model_results,
        exposure_vars=exposure_vars
    )
    fig3.savefig(os.path.join(args.output_dir, "model_coefficients.png"))
    print(f"Model coefficients plot saved to {os.path.join(args.output_dir, 'model_coefficients.png')}")

    print("Analysis complete!")


def run_download(args):
    """Download NHANES data."""
    data.download_nhanes_data(
        output_dir=args.output_dir,
        id=args.id,
        filename=args.filename,
        direct_url=args.direct_url
    )


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
