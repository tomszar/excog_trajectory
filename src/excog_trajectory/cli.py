#!/usr/bin/env python3
"""
Command-line interface for the excog_trajectory package.

This module provides CLI commands for analyzing exposomic trajectories
of cognitive decline in NHANES data and for downloading NHANES data.
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from excog_trajectory.visualization_theme import apply_theme
from excog_trajectory.preprocess import standardize_targets

from excog_trajectory import analysis, columns, data, trajectory, visualization, constants, schema
from excog_trajectory.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze exposomic trajectories of cognitive decline in NHANES"
    )

    # Helper types with validation
    def positive_int(val: str) -> int:
        iv = int(val)
        if iv <= 0:
            raise argparse.ArgumentTypeError("Value must be a positive integer")
        return iv

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Parser for the 'clean' command
    clean_parser = subparsers.add_parser(
        "clean", help="Clean and prepare NHANES data for analysis"
    )
    clean_io = clean_parser.add_argument_group("I/O")
    clean_io.add_argument(
        "--output-dir",
        type=str,
        default=constants.DEFAULT_CLEAN_OUTPUT_DIR,
        help="Directory to save results and figures",
    )
    clean_io.add_argument(
        "--output-data",
        type=str,
        default=constants.DEFAULT_CLEAN_OUTPUT_DATA,
        help="Directory to save processed data",
    )

    # Parser for the 'plsr' command
    plsr_parser = subparsers.add_parser(
        "plsr", help="Run Partial Least Squares Regression on NHANES data"
    )
    plsr_io = plsr_parser.add_argument_group("I/O")
    plsr_cv = plsr_parser.add_argument_group("Cross-Validation")
    plsr_opt = plsr_parser.add_argument_group("Options")
    plsr_io.add_argument(
        "--data-path",
        type=str,
        default=constants.DEFAULT_PLSR_DATA_PATH,
        help="Path to the directory containing imputed NHANES datasets",
    )
    plsr_io.add_argument(
        "--output-dir",
        type=str,
        default=constants.DEFAULT_PLSR_OUTPUT_DIR,
        help="Directory to save PLSR results",
    )
    plsr_opt.add_argument(
        "--scale",
        type=bool,
        default=True,
        help="Whether to standardize the data before running PLSR",
    )
    plsr_cv.add_argument(
        "--outer-folds",
        type=positive_int,
        default=8,
        help="Number of folds for the outer cross-validation loop",
    )
    plsr_cv.add_argument(
        "--inner-folds",
        type=positive_int,
        default=7,
        help="Number of folds for the inner cross-validation loop",
    )
    plsr_cv.add_argument(
        "--max-components",
        type=positive_int,
        default=5,
        help="Maximum number of components to try in cross-validation",
    )
    plsr_cv.add_argument(
        "--n-repetitions",
        type=positive_int,
        default=10,
        help="Number of times to repeat the cross-validation process",
    )
    plsr_cv.add_argument(
        "--random-state",
        type=positive_int,
        default=1203,
        help="Random state for cross-validation repeatability",
    )
    plsr_opt.add_argument(
        "--use-cache",
        type=bool,
        default=False,
        help="Reuse existing PLSR results table in each dataset output directory if present",
    )

    # Parser for the 'snf' command
    snf_parser = subparsers.add_parser(
        "snf", help="Run Similarity Network Fusion on NHANES data"
    )
    snf_io = snf_parser.add_argument_group("I/O")
    snf_opt = snf_parser.add_argument_group("Options")

    # Parser for the 'trajectory' command
    trajectory_parser = subparsers.add_parser(
        "trajectory",
        help="Compare cognitive decline trajectories between males and females across DSST categories",
    )
    traj_io = trajectory_parser.add_argument_group("I/O")
    traj_io.add_argument(
        "--data-path",
        type=str,
        default=constants.DEFAULT_TRAJECTORY_DATA_PATH,
        help="Path to the imputed data file used in the PLSR analysis",
    )
    traj_io.add_argument(
        "--model-path",
        type=str,
        default=constants.DEFAULT_TRAJECTORY_MODEL_PATH,
        help="Path to the saved PLSR model",
    )
    traj_io.add_argument(
        "--output-dir",
        type=str,
        default=constants.DEFAULT_TRAJECTORY_OUTPUT_DIR,
        help="Directory to save trajectory comparison results and plots",
    )
    snf_io.add_argument(
        "--data-path",
        type=str,
        default=constants.DEFAULT_SNF_DATA_PATH,
        help="Path to the imputed NHANES dataset",
    )
    snf_io.add_argument(
        "--output-dir",
        type=str,
        default=constants.DEFAULT_SNF_OUTPUT_DIR,
        help="Directory to save SNF results",
    )
    snf_opt.add_argument(
        "--k",
        type=positive_int,
        default=20,
        help="Number of nearest neighbors to consider",
    )
    snf_opt.add_argument(
        "--t",
        type=positive_int,
        default=20,
        help="Number of iterations for the fusion process",
    )
    snf_opt.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Parameter controlling the importance of local vs. global structure",
    )
    snf_opt.add_argument(
        "--scale",
        type=bool,
        default=True,
        help="Whether to standardize the data before running SNF",
    )

    # Parser for the 'download' command
    download_parser = subparsers.add_parser("download", help="Download NHANES data")
    dl_io = download_parser.add_argument_group("I/O")
    dl_io.add_argument(
        "--output-dir",
        type=str,
        default=constants.DEFAULT_DOWNLOAD_OUTPUT_DIR,
        help="Directory to save downloaded data",
    )
    dl_io.add_argument(
        "--direct-url",
        type=str,
        nargs="+",
        default=constants.DEFAULT_DIRECT_URLS,
        help="Direct URL(s) to download the data from. Can provide multiple URLs.",
    )
    dl_io.add_argument(
        "--filename",
        type=str,
        default=constants.DEFAULT_NHANES_FILENAME,
        help="Name to save the downloaded file as",
    )

    # Parser for the 'impute' command
    impute_parser = subparsers.add_parser(
        "impute", help="Impute missing values in NHANES data using MICE"
    )
    imp_io = impute_parser.add_argument_group("I/O")
    imp_opt = impute_parser.add_argument_group("Options")
    imp_io.add_argument(
        "--data-path",
        type=str,
        default=constants.DEFAULT_IMPUTE_DATA_PATH,
        help="Path to the cleaned NHANES dataset",
    )
    imp_io.add_argument(
        "--output-path",
        type=str,
        default=constants.DEFAULT_IMPUTE_OUTPUT_PATH,
        help="Path to save the imputed dataset",
    )
    imp_opt.add_argument(
        "--n-imputations",
        type=positive_int,
        default=5,
        help="Number of imputations to perform",
    )
    imp_opt.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility",
    )
    imp_opt.add_argument(
        "--n-iterations",
        type=positive_int,
        default=5,
        help="Number of iterations for the imputation procedure",
    )
    imp_opt.add_argument(
        "--save-kernel",
        type=bool,
        default=False,
        help="Whether to save the imputation kernel for future use",
    )
    imp_opt.add_argument(
        "--load-kernel",
        type=str,
        default=None,
        help="Path to load an existing imputation kernel from. If provided, this will skip the imputation step.",
    )
    imp_opt.add_argument(
        "--diagnostic-plots",
        type=bool,
        default=False,
        help="Whether to generate diagnostic plots for the imputation process",
    )

    args = parser.parse_args()

    # If no command is specified, show help and exit
    if args.command is None:
        parser.print_help()
        exit(1)

    return args


def clean_data(args):
    """Clean and prepare NHANES data for analysis."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_data, exist_ok=True)

    print(f"Loading NHANES data...")
    nhanes_data = data.load_nhanes_data()

    # Define variables for analysis using the cols module
    cognitive_vars = columns.COGNITIVE_VARS  # Cognitive function right responses
    cognitive_cats = columns.COGNITIVE_CAT  # Cognitive categories
    covariates = columns.COVARIATES  # Demographics and survey cycle
    cols_to_drop_na = cognitive_vars + covariates
    cols_to_drop = columns.COLS_TO_DROP  # Columns to drop

    print("Removing NaN values and unnecessary cols...")
    for dat in nhanes_data:
        for col in cols_to_drop_na:
            if col in nhanes_data[dat].columns:
                nhanes_data[dat] = data.remove_nan_from_columns(nhanes_data[dat], col)
        for col2 in cols_to_drop:
            if col2 in nhanes_data[dat].columns:
                nhanes_data[dat] = nhanes_data[dat].drop(columns=col2)

    # Apply QC rules to each dataset separately
    print("Applying QC rules to each dataset separately...")
    for dat in nhanes_data:
        print(f"Applying QC rules to {dat}...")
        nhanes_data[dat] = data.apply_qc_rules(
            nhanes_data[dat],
            cognitive_vars,
            covariates=covariates,
            standardize=True,
            log2_transform=True,
        )
        data.categorize_dsst_by_age(nhanes_data[dat])
        # Save the individual cleaned datasets
        output_file = os.path.join(args.output_data, f"cleaned_nhanes_{dat}.csv")
        nhanes_data[dat].to_csv(output_file, index=True)
        print(f"Cleaned {dat} saved to {output_file}")

        # Calculate percentage of missing data for each individual dataset
        print(f"Calculating percentage of missing data for {dat}...")
        missing_data_df = data.get_percentage_missing(nhanes_data[dat])

        # Save the missing data percentages for each individual dataset
        missing_file = os.path.join(args.output_data, f"percentage_missing_{dat}.csv")
        missing_data_df.to_csv(missing_file, index=False)
        print(f"Percentage of missing data for {dat} saved to {missing_file}")

        # Create correlation matrix of exposure variables for each individual dataset
        print(f"Creating correlation matrix of exposure variables for {dat}...")
        # Get exposure variables using the cols module
        exposure_vars = columns.get_exposure_vars(
            data=nhanes_data[dat],
            cognitive_vars=cognitive_vars + cognitive_cats,
            covariates=covariates,
        )

        visualization.plot_exposure_correlation_matrix(
            data=nhanes_data[dat][exposure_vars],
            fname=os.path.join(
                args.output_dir, f"exposure_correlation_matrix_{dat}.png"
            ),
        )
        print(
            f"Exposure correlation matrix for {dat} saved to {os.path.join(args.output_dir, f'exposure_correlation_matrix_{dat}.png')}"
        )

    # Combine data from both files after applying QC rules
    print("Combining data from multiple files...")

    # Ensure CFDDS and CFDRIGHT are treated as the same column in the combined dataset
    if (
        "CFDDS" in nhanes_data["data_1"].columns
        and "CFDRIGHT" in nhanes_data["data_2"].columns
    ):
        print(
            "Renaming CFDRIGHT to CFDDS in data_2 to treat them as the same column..."
        )
        nhanes_data["data_2"] = nhanes_data["data_2"].rename(
            columns={"CFDRIGHT": "CFDDS"}
        )
        # Update cognitive_vars list to reflect the renamed column
        if "CFDRIGHT" in cognitive_vars:
            cognitive_vars = [
                "CFDDS" if var == "CFDRIGHT" else var for var in cognitive_vars
            ]
    elif (
        "CFDRIGHT" in nhanes_data["data_1"].columns
        and "CFDDS" in nhanes_data["data_2"].columns
    ):
        print(
            "Renaming CFDDS to CFDRIGHT in data_2 to treat them as the same column..."
        )
        nhanes_data["data_2"] = nhanes_data["data_2"].rename(
            columns={"CFDDS": "CFDRIGHT"}
        )
        # Update cognitive_vars list to reflect the renamed column
        if "CFDDS" in cognitive_vars:
            cognitive_vars = [
                "CFDRIGHT" if var == "CFDDS" else var for var in cognitive_vars
            ]

    # First, perform an outer merge to get all cols from both dataframes
    combined_data = pd.merge(
        nhanes_data["data_1"],
        nhanes_data["data_2"],
        left_index=True,
        right_index=True,
        how="outer",
        suffixes=("_1", "_2"),
    )

    # Identify cols that have suffixes (indicating they were in both dataframes)
    suffix_1_cols = [col for col in combined_data.columns if col.endswith("_1")]
    base_cols = [
        col[:-2] for col in suffix_1_cols
    ]  # Remove the suffix to get the base column name

    # For each pair of suffixed cols, combine them into a single column
    for base_col in base_cols:
        col_1 = f"{base_col}_1"
        col_2 = f"{base_col}_2"

        # Create a new column that takes values from col_1, but uses col_2 where col_1 is NaN
        combined_data[base_col] = combined_data[col_1].combine_first(
            combined_data[col_2]
        )

        # Drop the original suffixed cols
        combined_data = combined_data.drop([col_1, col_2], axis=1)

    print(f"Combined data shape: {combined_data.shape}")

    # Filter cols in the combined dataset to keep only those with at least one observation in each Cycle
    print("Filtering cols to keep those with at least one observation in each Cycle...")

    # Check if we have the original 'Cycle' column or dummy variables
    cycle_dummy_cols = [
        col for col in combined_data.columns if col.startswith("Cycle_")
    ]

    if "Cycle" in combined_data.columns:
        # Original Cycle column exists, use it for grouping
        print("Using original Cycle column for filtering...")
        cycles = combined_data["Cycle"].unique()
        columns_to_keep = []

        for column in combined_data.columns:
            has_observation_in_all_cycles = True
            for cycle in cycles:
                cycle_data = combined_data[combined_data["Cycle"] == cycle]
                if (
                    cycle_data[column].isna().all()
                ):  # Check if ALL values are missing in this cycle
                    has_observation_in_all_cycles = False
                    break

            if has_observation_in_all_cycles:
                columns_to_keep.append(column)
    elif cycle_dummy_cols:
        # Cycle has been converted to dummy variables, use them for grouping
        print(f"Using Cycle dummy variables for filtering: {cycle_dummy_cols}")
        # Add zeros to dummy cols instead of NaNs
        combined_data[cycle_dummy_cols] = combined_data[cycle_dummy_cols].fillna(0)
        columns_to_keep = []

        for column in combined_data.columns:
            has_observation_in_all_cycles = True
            for cycle_col in cycle_dummy_cols:
                cycle_data = combined_data[combined_data[cycle_col] == True]
                # Check if there's at least one non-NaN value in this cycle for this column
                if len(cycle_data) > 0 and not cycle_data[column].notna().any():
                    has_observation_in_all_cycles = False
                    break

            if has_observation_in_all_cycles:
                columns_to_keep.append(column)
    else:
        # No Cycle information available, keep all cols
        print("Warning: No Cycle column or dummy variables found. Keeping all cols.")
        columns_to_keep = combined_data.columns.tolist()

    # Keep only cols with at least one observation in each Cycle
    combined_data = combined_data[columns_to_keep]
    print(f"Combined data shape after filtering: {combined_data.shape}")

    # Validate post-clean schema before saving
    try:
        schema.validate_post_clean_schema(combined_data)
    except Exception as _e:
        print(f"Warning: post-clean schema validation warning: {_e}")

    # Save the cleaned data
    combined_data.to_csv(
        os.path.join(args.output_data, "cleaned_nhanes.csv"), index=True
    )
    print(
        f"Cleaned data saved to {os.path.join(args.output_data, 'cleaned_nhanes.csv')}"
    )

    # Calculate percentage of missing data for each column in the combined dataset
    print("Calculating percentage of missing data for combined dataset...")
    missing_data_df = data.get_percentage_missing(combined_data)

    # Save the missing data percentages for the combined dataset
    missing_data_df.to_csv(
        os.path.join(args.output_data, "percentage_missing.csv"), index=False
    )
    print(
        f"Percentage of missing data saved to {os.path.join(args.output_data, 'percentage_missing.csv')}"
    )

    print("Creating visualizations for combined dataset...")
    exposure_vars = columns.get_exposure_vars(
        data=combined_data,
        cognitive_vars=cognitive_vars + cognitive_cats,
        covariates=covariates,
    )
    # Plot exposure distributions
    fig1 = visualization.plot_distributions(
        data=combined_data,
        vars=["CFDDS"] + exposure_vars,
        split_by_sex=True,
        save_path=args.output_dir,
        figsize=(10, 20),
    )
    print(
        f"Exposure distributions plot saved to {os.path.join(args.output_dir, 'distributions.png')}"
    )

    # Create correlation matrix of exposure variables for the combined dataset
    print("Creating correlation matrix of exposure variables for combined dataset...")
    # Get exposure variables using the cols module
    visualization.plot_exposure_correlation_matrix(
        data=combined_data[exposure_vars],
        fname=os.path.join(args.output_dir, "exposure_correlation_matrix.png"),
    )
    print(
        f"Exposure correlation matrix saved to {os.path.join(args.output_dir, 'exposure_correlation_matrix.png')}"
    )

    print("Analysis complete!")


def run_download(args):
    """Download NHANES data to the output directory."""
    # Download the data
    csv_paths = data.download_nhanes_data(
        output_dir=args.output_dir, filename=args.filename, direct_url=args.direct_url
    )

    # Handle both single path and list of paths
    if isinstance(csv_paths, list):
        print(f"Downloaded {len(csv_paths)} files")
        for path in csv_paths:
            print(f"  - {path}")
    else:
        # Convert single path to list for consistent handling
        csv_paths = [csv_paths]
        print(f"Downloaded 1 file: {csv_paths[0]}")

    print(f"NHANES data successfully downloaded to {args.output_dir}")


def run_imputation(args):
    """Run the imputation procedure."""

    print(f"Running imputation procedure...")
    output_path = args.output_path
    if output_path is None:
        output_path = "data/processed/imputed/"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Call the impute_exposure_variables function
    kernel = data.impute_exposure_variables(
        data_path=args.data_path,
        output_path=output_path,
        n_imputations=args.n_imputations,
        random_state=args.random_state,
        n_iterations=args.n_iterations,
        save_kernel=args.save_kernel,
        load_kernel=args.load_kernel,
        diagnostic_plots=args.diagnostic_plots,
    )

    # Create output directory for correlation matrices if it doesn't exist
    correlation_output_dir = "results/correlation_matrices"
    os.makedirs(correlation_output_dir, exist_ok=True)

    # Create correlation matrices for each imputed dataset
    print(f"Creating correlation matrices for each imputed dataset...")
    for i in range(args.n_imputations):
        dataset_num = i + 1
        filename = f"imputed_nhanes_dat{dataset_num}.csv"
        filepath = os.path.join(output_path, filename)

        print(f"Processing imputed dataset {dataset_num}...")

        # Load the imputed dataset
        imputed_data = pd.read_csv(filepath)
        exposure_vars = columns.get_exposure_vars(imputed_data)

        # Create correlation matrix
        print(f"Creating correlation matrix for dataset {dataset_num}...")
        visualization.plot_exposure_correlation_matrix(
            data=imputed_data[exposure_vars],
            fname=os.path.join(
                correlation_output_dir,
                f"exposure_correlation_matrix_dataset{dataset_num}.png",
            ),
            dpi=300,
        )


def run_plsr_analysis(args):
    """Run the PLSR analysis pipeline.

    This function processes multiple imputed datasets from the specified directory,
    runs PLSR on each dataset, and saves the results in separate subdirectories.
    For each imputed dataset, it:
    1. Creates a subdirectory under the output directory
    2. Runs PLSR with double cross-validation
    3. Saves the PLSR results table as a CSV
    4. Creates and saves the best model
    5. Calculates and saves VIP-X scores
    6. Creates and saves visualizations of VIP-X scores and PLSR scores

    After processing all datasets, it determines the most common number of LVs across all datasets,
    creates a single best model with that number, and saves it to the main output directory.
    """
    # Create main output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Check if data_path is a directory
    if not os.path.isdir(args.data_path):
        print(
            f"Error: {args.data_path} is not a directory. Please provide a directory containing imputed datasets."
        )
        return

    # Find all imputed dataset files in the directory
    imputed_files = [
        f
        for f in os.listdir(args.data_path)
        if f.startswith("imputed_nhanes_dat") and f.endswith(".csv")
    ]

    if not imputed_files:
        print(f"Error: No imputed dataset files found in {args.data_path}")
        return

    print(f"Found {len(imputed_files)} imputed dataset files in {args.data_path}")

    # List to store PLSR results tables from all imputed datasets
    all_plsr_tables = []

    # List xs and ys
    exes = []
    eyes = []

    # Process each imputed dataset
    for imputed_file in sorted(imputed_files):
        dataset_num = imputed_file.split("dat")[1].split(".")[0]
        print(f"\nProcessing imputed dataset {dataset_num}...")

        # Create a subdirectory for this imputed dataset
        dataset_output_dir = os.path.join(args.output_dir, f"dataset{dataset_num}")
        os.makedirs(dataset_output_dir, exist_ok=True)

        # Load the imputed dataset
        file_path = os.path.join(args.data_path, imputed_file)
        print(f"Loading imputed NHANES data from {file_path}...")
        data_df = pd.read_csv(file_path, index_col=0)

        # Define variables for analysis using the cols module
        cognitive_vars = (
            columns.COGNITIVE_VARS
        )  # Using a specific cognitive variable for PLSR
        covariates = columns.COVARIATES  # Demographics and survey cycle
        cognitive_cats = columns.COGNITIVE_CAT  # Cognitive categories

        # Validate cols exist in the dataset
        valid_cognitive_vars = columns.validate_columns(
            data_df, cognitive_vars, raise_error=False
        )
        if not valid_cognitive_vars:
            columns.require_columns(
                data_df, cognitive_vars, context="PLSR analysis (cognitive variables)"
            )
        valid_covariates = columns.validate_columns(
            data_df, covariates, raise_error=False
        )
        dummy_vars = columns.get_dummy_vars(data_df)

        # Get exposure variables using the cols module
        exposure_vars = columns.get_exposure_vars(
            data=data_df,
            cognitive_vars=valid_cognitive_vars + cognitive_cats,
            covariates=valid_covariates,
        )
        x = data_df[exposure_vars]

        # Get the cognitive variables
        y_original = data_df[valid_cognitive_vars]
        y, _ = standardize_targets(y_original)

        # Append x and y
        exes.append(x)
        eyes.append(y)

        print(
            f"Running PLSR with {len(exposure_vars)} exposure variables, and "
            f"{len(valid_cognitive_vars)} cognitive variables..."
        )

        if args.max_components > x.shape[1]:
            print(
                f"Warning: Number of components ({args.max_components}) is greater than the number of variables ({x.shape[1]}). "
                f"Setting max_components to {x.shape[1]}."
            )
            args.max_components = x.shape[1]

        if args.n_repetitions > 1:
            print(
                f"Running PLSR with double cross-validation ({args.outer_folds} outer folds, "
                f"{args.inner_folds} inner folds) repeated {args.n_repetitions} times..."
            )
        else:
            print(
                f"Running PLSR with double cross-validation ({args.outer_folds} outer folds, "
                f"{args.inner_folds} inner folds)..."
            )

        # Optionally reuse cached CV results table
        plsr_table_path = os.path.join(dataset_output_dir, "plsr_results_table.csv")
        plsr_table = None
        if getattr(args, "use_cache", False) and os.path.exists(plsr_table_path):
            try:
                plsr_table = pd.read_csv(plsr_table_path)
                print(f"Loaded cached PLSR results table from {plsr_table_path}")
            except Exception:
                plsr_table = None
        if plsr_table is None:
            plsr_results = analysis.pls_double_cv(
                x=x,
                y=y,
                n_repeats=args.n_repetitions,
                max_components=args.max_components,
                cv2_splits=args.outer_folds,
                cv1_splits=args.inner_folds,
                random_state=args.random_state,
            )
            plsr_table = plsr_results["table"]
            # Save table for this dataset
            plsr_table.to_csv(plsr_table_path, index=False)
        # Append the PLSR results table to our list of all tables
        all_plsr_tables.append(plsr_table)

        # Append the PLSR results table to our list of all tables
        # (already appended above, possibly from cache)

        # Print information about the final model for this dataset
        mode = int(plsr_table["LV"].mode()[0])
        print(
            f"\nThe most repeated number of LV for dataset {dataset_num}: {str(mode)}"
        )
        from sklearn.cross_decomposition import PLSRegression

        best_model = PLSRegression(
            n_components=mode,
            scale=False,  # Changed from True to False as we're handling scaling separately
            max_iter=1000,
        ).fit(X=x, y=y)
        print(
            f"A final model has been trained on dataset {dataset_num} using {str(mode)} components."
        )

        # Save the best model for this dataset
        with open(os.path.join(dataset_output_dir, "best_model.pkl"), "wb") as f:
            pickle.dump(best_model, f)

        # Ensure cognitive categories (DSST_High, DSST_Average, DSST_Low) are present in the data
        # Check if cognitive categories exist in the data
        missing_categories = [
            cat for cat in cognitive_cats if cat not in data_df.columns
        ]

        if missing_categories:
            print(
                f"Categorizing DSST scores by age to create cognitive categories: {', '.join(cognitive_cats)}"
            )
            data_df = data.categorize_dsst_by_age(data_df)

        # Calculate VIP-X scores
        print(f"Calculating VIP-X scores for dataset {dataset_num}...")
        vip_x_scores = analysis.calculate_vip_x_scores(best_model, x)

        # Save VIP-X scores to CSV
        vip_x_file = os.path.join(dataset_output_dir, "vip_x_scores.csv")
        vip_x_scores.to_csv(vip_x_file)
        print(f"VIP-X scores saved to {vip_x_file}")

        # Create visualization of VIP-X scores
        visualization.plot_vip_x_scores(
            vip_x_scores=vip_x_scores, output_dir=dataset_output_dir
        )

        # Create scatter plots of the first two columns of x_scores
        visualization.plot_plsr_scores(
            best_model=best_model,
            data_df=data_df,
            cognitive_vars=exposure_vars + valid_cognitive_vars,
            output_dir=dataset_output_dir,
        )


        print(
            f"PLSR results for dataset {dataset_num} saved to {os.path.join(dataset_output_dir, 'best_model.pkl')}"
        )

    # Concatenate all PLSR results tables
    if all_plsr_tables:
        # List to store best models from each imputed dataset
        all_best_models = []

        all_tables = pd.concat(all_plsr_tables, ignore_index=True)

        # Determine the most common number of LVs across all datasets
        best_lv = int(all_tables["LV"].mode()[0])
        print(f"\nThe most common number of LVs across all datasets: {best_lv}")

        # Create a single best model with the most common number of LVs using the last processed dataset
        from sklearn.cross_decomposition import PLSRegression

        for x, y in zip(exes, eyes):
            # Create a PLSRegression model with the best number of components
            print(f"Creating best model with {best_lv} components...")
            best_model = PLSRegression(
                n_components=best_lv, scale=False, max_iter=1000
            ).fit(X=x, y=y)
            all_best_models.append(best_model)

        # Save the single best model to the main output directory
        with open(os.path.join(args.output_dir, "best_model.pkl"), "wb") as f:
            pickle.dump(all_best_models, f)

        print(
            f"\nBest model with {best_lv} components saved to {os.path.join(args.output_dir, 'best_model.pkl')}"
        )
    else:
        print("\nNo PLSR results tables were generated. No best model was created.")

    print("PLSR analysis complete for all imputed datasets!")


def run_snf_analysis(args):
    """Run the SNF analysis pipeline."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading imputed NHANES data from {args.data_path}...")
    data_df = pd.read_csv(args.data_path)

    # Define variables for analysis
    cognitive_vars = ["CFDRIGHT"]  # Cognitive function right responses
    # Note: SNF uses a different set of covariates than other functions
    covariates = [
        "RIDAGEYR",
        "female",
        "male",
        "black",
        "mexican",
        "other_hispanic",
        "other_eth",
        "SES_LEVEL",
        "education",
        "SDDSRVYR",
    ]  # Demographics and survey cycle

    # Validate cols exist in the dataset
    valid_cognitive_vars = columns.validate_columns(
        data_df, cognitive_vars, raise_error=False
    )
    valid_covariates = columns.validate_columns(data_df, covariates, raise_error=False)

    # Create exposure categories based on column name patterns using the cols module
    print("Creating exposure categories based on column name patterns...")
    exposure_categories = columns.categorize_exposure_vars(data_df)

    print(f"Identified {len(exposure_categories)} exposure categories")
    for category, vars_list in exposure_categories.items():
        print(f"  {category}: {len(vars_list)} variables")

    print(
        f"Running SNF with {len(exposure_categories)} exposure categories, {len(cognitive_vars)} cognitive variables, and {len(covariates)} covariates..."
    )

    # Run SNF
    from excog_trajectory import analysis

    snf_results = analysis.run_snf(
        data=data_df,
        exposure_categories=exposure_categories,
        cognitive_vars=valid_cognitive_vars,
        covariates=valid_covariates,
        k=args.k,
        t=args.t,
        alpha=args.alpha,
        scale=args.scale,
    )

    # Save the results
    import pickle

    with open(os.path.join(args.output_dir, "snf_results.pkl"), "wb") as f:
        pickle.dump(snf_results, f)

    print(f"SNF results saved to {os.path.join(args.output_dir, 'snf_results.pkl')}")

    # Create visualizations
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE

    # Plot the fused similarity matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(snf_results["fused_matrix"], cmap="viridis")
    plt.colorbar(label="Similarity")
    plt.title("SNF Fused Similarity Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "snf_fused_matrix.png"), dpi=300)

    # Apply t-SNE to the fused similarity matrix for visualization
    print("Applying t-SNE to the fused similarity matrix...")
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(snf_results["fused_matrix"])

    # Apply K-means clustering to the t-SNE result
    n_clusters = 3  # Can be adjusted
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(tsne_result)

    # Plot the t-SNE result with cluster labels
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        tsne_result[:, 0],
        tsne_result[:, 1],
        c=cluster_labels,
        cmap="viridis",
        alpha=0.8,
    )
    plt.colorbar(scatter, label="Cluster")
    plt.title("t-SNE Visualization of SNF Fused Similarity Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "snf_tsne.png"), dpi=300)

    print(f"SNF visualizations saved to {args.output_dir}")
    print("SNF analysis complete!")


def run_trajectory_comparison(args):
    """Run trajectory comparison analysis."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading data from {args.data_path}...")
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        return

    # Load data and set it up
    with open(args.model_path, "rb") as f:
        model = pickle.load(f)
    all_xscores = []
    all_loadings = []
    for mod in model:
        all_xscores.append(mod.x_scores_)
        all_loadings.append(mod.x_loadings_)
    x_scores = pd.DataFrame(np.mean(np.stack(all_xscores, axis=0), axis=0))
    x_loadings = pd.DataFrame(np.mean(np.stack(all_loadings, axis=0), axis=0))
    x_scores.columns = [f"LV{i+1}" for i in range(x_scores.shape[1])]
    df = pd.read_csv(args.data_path)
    df_full = pd.concat([df, x_scores], axis=1)
    df_full.set_index("sample", inplace=True)

    covariates = columns.COVARIATES
    valid_covariates = columns.validate_columns(df, covariates, raise_error=False)
    covariates_cat = columns.CATEGORICAL_COVARIATES
    cognitive_cat = columns.COGNITIVE_CAT
    # Cognitive outcome variables used in PLSR
    cognitive_vars = columns.COGNITIVE_VARS
    valid_cognitive_vars = columns.validate_columns(df, cognitive_vars, raise_error=False)
    dummy_vars = columns.get_dummy_vars(df_full, covariates_cat)

    # Create initial model matrix
    initial_model_matrix = df[cognitive_cat + dummy_vars + valid_covariates]

    # Create proper model matrix using the new function
    model_matrix = trajectory.create_model_matrix(
        model_matrix=initial_model_matrix,
        cognitive_cat=cognitive_cat,
        dummy_vars=dummy_vars,
        valid_covariates=valid_covariates,
        add_interactions=True,
    )
    reduced_model = trajectory.create_model_matrix(
        model_matrix=initial_model_matrix,
        cognitive_cat=cognitive_cat,
        dummy_vars=dummy_vars,
        valid_covariates=valid_covariates,
        add_interactions=False,
    )

    y = x_scores
    betas = trajectory.estimate_betas(model_matrix, y)
    ls_vectors = trajectory._get_ls_vectors(model_matrix)
    contrast = [[0, 1, 2], [3, 4, 5]]
    obs_vect = pd.DataFrame(np.matmul(ls_vectors, betas))
    # Set column names for obs_vect to match the LV columns in x_scores
    obs_vect.columns = [f"LV{i+1}" for i in range(obs_vect.shape[1])]
    obs_vect_transformed = obs_vect.copy()
    obs_vect_transformed.iloc[[0, 1, 2], :] -= obs_vect_transformed.iloc[1, :]
    obs_vect_transformed.iloc[[3, 4, 5], :] -= obs_vect_transformed.iloc[4, :]

    deltas, angles, shapes = trajectory.estimate_difference(
        y, model_matrix, ls_vectors, contrast
    )

    r_deltas, r_angles, r_shapes = trajectory.RRPP(
        y, model_matrix, reduced_model, ls_vectors, contrast, 9999
    )

    total_rep = 10000
    pvals = [
        (sum(r_angles > angles) / total_rep)[0, 1],
        (sum(r_deltas > deltas) / total_rep)[0, 1],
        (sum(r_shapes > shapes) / total_rep)[0, 1],
    ]

    # Export p-values to CSV
    pval_df = pd.DataFrame(
        {"comparison_type": ["orientation", "magnitude", "shape"], "pvalue": pvals}
    )
    pval_path = os.path.join(args.output_dir, "pvalues.csv")
    pval_df.to_csv(pval_path, index=False)
    print(f"P-values exported to {pval_path}")

    # Create cognitive trajectory plots using the new function in visualization.py
    plot_names = ["cognitive_trajectory", "cognitive_trajectory_transformed"]
    for i, vect in enumerate([obs_vect, obs_vect_transformed * 2]):
        visualization.plot_cognitive_trajectory(
            x_scores=x_scores,
            obs_vect=vect,
            output_dir=args.output_dir,
            filename=plot_names[i],
        )

    print(f"Trajectory plots saved to {args.output_dir}")

    # Create a multi-panel PLSR biplot at the trajectory stage
    try:
        # Reconstruct the X matrix used by the model for correct variable ordering/names
        X = df[model[0].feature_names_in_]
        # Compute VIP-X scores for coloring
        vip_x_scores = analysis.calculate_vip_x_scores(model[0], X)
        visualization.plot_plsr_biplot(
            model=model[0],
            X=X,
            outcome_names=valid_cognitive_vars,
            output_dir=args.output_dir,
            vip_scores=vip_x_scores,
            filename="plsr_biplot_all_components.png",
        )
        # Also create the 3D biplot for the first three components
        visualization.plot_plsr_biplot_3d(
            model=model[0],
            X=X,
            outcome_names=valid_cognitive_vars,
            output_dir=args.output_dir,
            filename="plsr_biplot_3d_components_1_2_3.png",
        )
        print(f"PLSR biplots (2D and 3D) saved to {args.output_dir}")
    except Exception as e:
        print(f"Warning: Failed to create PLSR biplots at trajectory stage: {e}")

    # Transform vectors back to original X matrix values and export to CSV
    original_x = trajectory.transform_vectors_to_original(obs_vect,
                                                          x_loadings=x_loadings,
                                                          feature_names=model[0].feature_names_in_)
    original_x_path = os.path.join(args.output_dir, "original_x_values.csv")
    original_x.to_csv(original_x_path, index=False)
    original_x = trajectory.transform_vectors_to_original(obs_vect_transformed,
                                                          x_loadings=x_loadings,
                                                          feature_names=model[0].feature_names_in_)
    original_x_path = os.path.join(args.output_dir, "original_x_values_transformed.csv")
    original_x.to_csv(original_x_path, index=False)
    print(f"Original X matrix values saved to {original_x_path}")


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    # Apply a consistent plotting theme once
    try:
        apply_theme()
    except Exception:
        pass

    # Execute the appropriate command
    if args.command == "clean":
        clean_data(args)
    elif args.command == "download":
        run_download(args)
    elif args.command == "impute":
        run_imputation(args)
    elif args.command == "plsr":
        run_plsr_analysis(args)
    elif args.command == "snf":
        run_snf_analysis(args)
    elif args.command == "trajectory":
        run_trajectory_comparison(args)
    else:
        print(f"Unknown command: {args.command}")
        exit(1)


if __name__ == "__main__":
    main()
