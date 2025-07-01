#!/usr/bin/env python3
"""
Command-line interface for the excog_trajectory package.

This module provides CLI commands for analyzing exposomic trajectories
of cognitive decline in NHANES data and for downloading NHANES data.
"""

import argparse
import os

import pandas as pd

from excog_trajectory import data, visualization


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze exposomic trajectories of cognitive decline in NHANES"
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Parser for the 'clean' command
    clean_parser = subparsers.add_parser(
        "clean", help="Clean and prepare NHANES data for analysis"
    )

    # Parser for the 'plsr' command
    plsr_parser = subparsers.add_parser(
        "plsr", help="Run Partial Least Squares Regression on NHANES data"
    )
    plsr_parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/imputed_nhanes_dat1.csv",
        help="Path to the imputed NHANES dataset",
    )
    plsr_parser.add_argument(
        "--output-dir",
        type=str,
        default="results/plsr",
        help="Directory to save PLSR results",
    )
    plsr_parser.add_argument(
        "--scale",
        type=bool,
        default=True,
        help="Whether to standardize the data before running PLSR",
    )
    plsr_parser.add_argument(
        "--outer-folds",
        type=int,
        default=8,
        help="Number of folds for the outer cross-validation loop",
    )
    plsr_parser.add_argument(
        "--inner-folds",
        type=int,
        default=7,
        help="Number of folds for the inner cross-validation loop",
    )
    plsr_parser.add_argument(
        "--max-components",
        type=int,
        default=5,
        help="Maximum number of components to try in cross-validation",
    )
    plsr_parser.add_argument(
        "--n-repetitions",
        type=int,
        default=10,
        help="Number of times to repeat the cross-validation process",
    )

    # Parser for the 'snf' command
    snf_parser = subparsers.add_parser(
        "snf", help="Run Similarity Network Fusion on NHANES data"
    )
    snf_parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/imputed_nhanes_dat1.csv",
        help="Path to the imputed NHANES dataset",
    )
    snf_parser.add_argument(
        "--output-dir",
        type=str,
        default="results/snf",
        help="Directory to save SNF results",
    )
    snf_parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Number of nearest neighbors to consider",
    )
    snf_parser.add_argument(
        "--t",
        type=int,
        default=20,
        help="Number of iterations for the fusion process",
    )
    snf_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Parameter controlling the importance of local vs. global structure",
    )
    snf_parser.add_argument(
        "--scale",
        type=bool,
        default=True,
        help="Whether to standardize the data before running SNF",
    )
    clean_parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results and figures",
    )
    clean_parser.add_argument(
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
        "--direct-url",
        type=str,
        nargs="+",
        default=["https://osf.io/download/9aupq/", "https://osf.io/download/9vewm/"],
        help="Direct URL(s) to download the data from. Can provide multiple URLs.",
    )
    download_parser.add_argument(
        "--filename",
        type=str,
        default="nhanes_data.csv",
        help="Name to save the downloaded file as",
    )

    # Parser for the 'impute' command
    impute_parser = subparsers.add_parser(
        "impute", help="Impute missing values in NHANES data using MICE"
    )
    impute_parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/cleaned_nhanes.csv",
        help="Path to the cleaned NHANES dataset",
    )
    impute_parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save the imputed dataset",
    )
    impute_parser.add_argument(
        "--n-imputations",
        type=int,
        default=5,
        help="Number of imputations to perform",
    )
    impute_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility",
    )
    impute_parser.add_argument(
        "--n-iterations",
        type=int,
        default=3,
        help="Number of iterations for the imputation procedure",
    )
    impute_parser.add_argument(
        "--n-random-vars",
        type=int,
        default=None,
        help="Number of random variables to select for imputation. If not provided, all variables are used.",
    )
    impute_parser.add_argument(
        "--save-kernel",
        type=bool,
        default=False,
        help="Whether to save the imputation kernel for future use",
    )
    impute_parser.add_argument(
        "--load-kernel",
        type=str,
        default=None,
        help="Path to load an existing imputation kernel from. If provided, this will skip the imputation step.",
    )
    impute_parser.add_argument(
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

    # Define variables for analysis
    cognitive_vars = ["CFDRIGHT", "CFDDS"]  # Cognitive function right responses
    covariates = ["Cycle", "RIDAGEYR", "RIAGENDR", "INDFMPIR",
                  "DMDEDUC2", "RIDRETH1"]  # Demographics and survey cycle
    cols_to_drop_na = cognitive_vars + covariates
    cols_to_drop = ["SDDSRVYR", "INDHHINC", "INDHHIN2"]  # Columns to drop

    print("Removing NaN values and unnecessary columns...")
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
        nhanes_data[dat] = data.apply_qc_rules(nhanes_data[dat], cognitive_vars,
                                               covariates, standardize=True,
                                               log2_transform=True)

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
        exposure_vars = [col for col in nhanes_data[dat].columns if col not in cognitive_vars + covariates]
        visualization.plot_exposure_correlation_matrix(
            data=nhanes_data[dat][exposure_vars],
            fname=os.path.join(args.output_dir, f"exposure_correlation_matrix_{dat}.png"),
        )
        print(f"Exposure correlation matrix for {dat} saved to {os.path.join(args.output_dir, f'exposure_correlation_matrix_{dat}.png')}")

    # Combine data from both files after applying QC rules
    print("Combining data from multiple files...")

    # Ensure CFDDS and CFDRIGHT are treated as the same column in the combined dataset
    if "CFDDS" in nhanes_data["data_1"].columns and "CFDRIGHT" in nhanes_data["data_2"].columns:
        print("Renaming CFDRIGHT to CFDDS in data_2 to treat them as the same column...")
        nhanes_data["data_2"] = nhanes_data["data_2"].rename(columns={"CFDRIGHT": "CFDDS"})
        # Update cognitive_vars list to reflect the renamed column
        if "CFDRIGHT" in cognitive_vars:
            cognitive_vars = ["CFDDS" if var == "CFDRIGHT" else var for var in cognitive_vars]
    elif "CFDRIGHT" in nhanes_data["data_1"].columns and "CFDDS" in nhanes_data["data_2"].columns:
        print("Renaming CFDDS to CFDRIGHT in data_2 to treat them as the same column...")
        nhanes_data["data_2"] = nhanes_data["data_2"].rename(columns={"CFDDS": "CFDRIGHT"})
        # Update cognitive_vars list to reflect the renamed column
        if "CFDDS" in cognitive_vars:
            cognitive_vars = ["CFDRIGHT" if var == "CFDDS" else var for var in cognitive_vars]

    # First, perform an outer merge to get all columns from both dataframes
    combined_data = pd.merge(nhanes_data["data_1"], nhanes_data["data_2"],
                             left_index=True, right_index=True, how="outer", suffixes=('_1', '_2'))

    # Identify columns that have suffixes (indicating they were in both dataframes)
    suffix_1_cols = [col for col in combined_data.columns if col.endswith('_1')]
    base_cols = [col[:-2] for col in suffix_1_cols]  # Remove the suffix to get the base column name

    # For each pair of suffixed columns, combine them into a single column
    for base_col in base_cols:
        col_1 = f"{base_col}_1"
        col_2 = f"{base_col}_2"

        # Create a new column that takes values from col_1, but uses col_2 where col_1 is NaN
        combined_data[base_col] = combined_data[col_1].combine_first(combined_data[col_2])

        # Drop the original suffixed columns
        combined_data = combined_data.drop([col_1, col_2], axis=1)

    print(f"Combined data shape: {combined_data.shape}")

    # Filter columns in the combined dataset to keep only those with at least one observation in each Cycle
    print("Filtering columns to keep those with at least one observation in each Cycle...")
    # Group by Cycle and check for observations in each column
    cycles = combined_data['Cycle'].unique()
    columns_to_keep = []

    for column in combined_data.columns:
        has_observation_in_all_cycles = True
        for cycle in cycles:
            cycle_data = combined_data[combined_data['Cycle'] == cycle]
            if cycle_data[column].isna().all():  # Check if ALL values are missing in this cycle
                has_observation_in_all_cycles = False
                break

        if has_observation_in_all_cycles:
            columns_to_keep.append(column)

    # Keep only columns with at least one observation in each Cycle
    combined_data = combined_data[columns_to_keep]
    print(f"Combined data shape after filtering: {combined_data.shape}")

    # Save the cleaned data
    combined_data.to_csv(os.path.join(args.output_data, "cleaned_nhanes.csv"), index=True)
    print(f"Cleaned data saved to {os.path.join(args.output_data, 'cleaned_nhanes.csv')}")

    # Calculate percentage of missing data for each column in the combined dataset
    print("Calculating percentage of missing data for combined dataset...")
    missing_data_df = data.get_percentage_missing(combined_data)

    # Save the missing data percentages for the combined dataset
    missing_data_df.to_csv(os.path.join(args.output_data, "percentage_missing.csv"), index=False)
    print(f"Percentage of missing data saved to {os.path.join(args.output_data, 'percentage_missing.csv')}")

    print("Creating visualizations for combined dataset...")
    # Plot exposure distributions
    fig1 = visualization.plot_distributions(
        data=combined_data,
        vars=cognitive_vars,
        save_path=args.output_dir,
    )
    print(f"Exposure distributions plot saved to {os.path.join(args.output_dir, 'distributions.png')}")

    # Create correlation matrix of exposure variables for the combined dataset
    print("Creating correlation matrix of exposure variables for combined dataset...")
    # Since we don't have description data, we'll use all columns except cognitive and covariates
    exposure_vars = [col for col in combined_data.columns if col not in cognitive_vars + covariates]
    visualization.plot_exposure_correlation_matrix(
        data=combined_data[exposure_vars],
        fname=os.path.join(args.output_dir, "exposure_correlation_matrix.png"),
    )
    print(f"Exposure correlation matrix saved to {os.path.join(args.output_dir, 'exposure_correlation_matrix.png')}")

    print("Analysis complete!")


def run_download(args):
    """Download NHANES data to the output directory."""
    # Download the data
    csv_paths = data.download_nhanes_data(
        output_dir=args.output_dir,
        filename=args.filename,
        direct_url=args.direct_url
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

    # Call the impute_exposure_variables function
    kernel = data.impute_exposure_variables(
        data_path=args.data_path,
        output_path=args.output_path,
        n_imputations=args.n_imputations,
        random_state=args.random_state,
        n_random_vars=args.n_random_vars,
        n_iterations=args.n_iterations,
        save_kernel=args.save_kernel,
        load_kernel=args.load_kernel,
        diagnostic_plots=args.diagnostic_plots,
    )

    # Set default output path if none is provided
    output_path = args.output_path
    if output_path is None:
        output_path = "data/processed/"

    # Note: We no longer have description data available

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

        # Create correlation matrix
        print(f"Creating correlation matrix for dataset {dataset_num}...")
        visualization.plot_exposure_correlation_matrix(
            data=imputed_data,
            fname=os.path.join(correlation_output_dir, f"exposure_correlation_matrix_dataset{dataset_num}.png"),
            dpi=300,
        )


def run_plsr_analysis(args):
    """Run the PLSR analysis pipeline."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading imputed NHANES data from {args.data_path}...")
    data_df = pd.read_csv(args.data_path)

    # Define variables for analysis
    cognitive_vars = ["CFDRIGHT"]  # Cognitive function right responses
    covariates = ["RIDAGEYR", "female", "male", "black", "mexican", "other_hispanic", "other_eth", "SES_LEVEL",
                  "education", "SDDSRVYR"]  # Demographics and survey cycle

    # Identify exposure variables (all variables that are not cognitive or covariates)
    all_vars = set(data_df.columns)
    non_exposure_vars = set(cognitive_vars + covariates + ["SEQN"])
    exposure_vars = list(all_vars - non_exposure_vars)

    print(f"Running PLSR with {len(exposure_vars)} exposure variables, {len(cognitive_vars)} cognitive variables, and {len(covariates)} covariates...")

    # Run PLSR with cross-validation (always enabled)
    from excog_trajectory import analysis

    if args.n_repetitions > 1:
        print(
            f"Running PLSR with double cross-validation ({args.outer_folds} outer folds, {args.inner_folds} inner folds) repeated {args.n_repetitions} times...")
    else:
        print(
            f"Running PLSR with double cross-validation ({args.outer_folds} outer folds, {args.inner_folds} inner folds)...")
    n_components_range = list(range(1, args.max_components + 1))
    plsr_results = analysis.run_plsr_cv(
        data=data_df,
        exposure_vars=exposure_vars,
        cognitive_vars=cognitive_vars,
        covariates=covariates,
        n_components_range=n_components_range,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        scale=args.scale,
        n_repetitions=args.n_repetitions,
    )

    # Print information about the final model
    if 'final_best_n_components' in plsr_results:
        final_best_n_comp = plsr_results['final_best_n_components']
        print(f"\nFinal best number of components: {final_best_n_comp}")
        print(f"A final model has been trained on the entire dataset using {final_best_n_comp} components.")

    # Save the results
    import pickle
    with open(os.path.join(args.output_dir, "plsr_results.pkl"), "wb") as f:
        pickle.dump(plsr_results, f)

    print(f"PLSR results saved to {os.path.join(args.output_dir, 'plsr_results.pkl')}")

    # Create visualizations
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    # Visualizations for cross-validation results

    # Plot distribution of best number of components
    plt.figure(figsize=(10, 6))
    best_components = plsr_results["best_n_components"]
    unique_components = sorted(set(best_components))
    counts = [best_components.count(comp) for comp in unique_components]
    plt.bar(unique_components, counts)
    plt.xlabel('Number of Components')
    plt.ylabel('Frequency')

    # Update title to reflect multiple repetitions if applicable
    if args.n_repetitions > 1:
        plt.title(
            f'Distribution of Optimal Number of Components Across {args.outer_folds} Outer Folds × {args.n_repetitions} Repetitions')
    else:
        plt.title('Distribution of Optimal Number of Components Across Outer Folds')

    plt.xticks(unique_components)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "plsr_cv_best_components.png"), dpi=300)

    # Highlight the final best number of components
    final_best_n_comp = plsr_results["final_best_n_components"]
    plt.figure(figsize=(10, 6))
    plt.bar(unique_components, counts,
            color=['blue' if comp != final_best_n_comp else 'red' for comp in unique_components])
    plt.xlabel('Number of Components')
    plt.ylabel('Frequency')
    plt.title(f'Final Best Number of Components: {final_best_n_comp}')
    plt.xticks(unique_components)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "plsr_cv_final_best_component.png"), dpi=300)

    # Plot outer fold AUROC scores
    plt.figure(figsize=(10, 6))
    outer_scores = plsr_results["outer_scores"]

    # If using multiple repetitions, we need to reshape the scores
    if args.n_repetitions > 1 and 'repetition_best_components' in plsr_results:
        # Calculate the number of outer folds
        n_outer_folds = len(plsr_results['repetition_best_components'][0])

        # Reshape the scores to get scores for each fold across repetitions
        reshaped_scores = np.array(outer_scores).reshape(args.n_repetitions, n_outer_folds)

        # Calculate mean and std for each fold across repetitions
        mean_scores = np.mean(reshaped_scores, axis=0)
        std_scores = np.std(reshaped_scores, axis=0)

        # Plot with error bars
        plt.bar(range(1, n_outer_folds + 1), mean_scores, yerr=std_scores, capsize=5)
        plt.axhline(y=np.mean(mean_scores), color='r', linestyle='-',
                    label=f'Mean AUROC: {np.mean(mean_scores):.3f}')
        plt.xlabel('Outer Fold')
        plt.ylabel('Mean AUROC Score (across repetitions)')
        plt.title(f'Mean AUROC Scores Across Outer Folds ({args.n_repetitions} Repetitions)')
        plt.xticks(range(1, n_outer_folds + 1))
    else:
        # Original plot for single repetition
        plt.bar(range(1, len(outer_scores) + 1), outer_scores)
        plt.axhline(y=np.mean(outer_scores), color='r', linestyle='-',
                    label=f'Mean AUROC: {np.mean(outer_scores):.3f}')
        plt.xlabel('Outer Fold')
        plt.ylabel('AUROC Score')
        plt.title('AUROC Scores Across Outer Folds')
        plt.xticks(range(1, len(outer_scores) + 1))

    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "plsr_cv_outer_scores.png"), dpi=300)

    # Plot inner fold AUROC scores for each number of components
    plt.figure(figsize=(12, 8))
    n_components_range = plsr_results["n_components_range"]
    inner_scores = plsr_results["inner_scores"]

    # Calculate mean scores for each number of components
    mean_scores = [np.mean(inner_scores[n_comp]) for n_comp in n_components_range]
    std_scores = [np.std(inner_scores[n_comp]) for n_comp in n_components_range]

    plt.errorbar(n_components_range, mean_scores, yerr=std_scores, fmt='o-', capsize=5)
    plt.xlabel('Number of Components')
    plt.ylabel('Mean AUROC Score')
    plt.title('Mean AUROC Scores Across Inner Folds by Number of Components')
    plt.xticks(n_components_range)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "plsr_cv_inner_scores.png"), dpi=300)

    # If there are multiple models (one per outer fold), we can visualize the loadings of the first component
    # from the model with the best performance on the outer test fold
    if plsr_results["models"]:
        best_model_idx = np.argmax(plsr_results["outer_scores"])
        best_model = plsr_results["models"][best_model_idx]

        # Plot X loadings for the best model from cross-validation
        plt.figure(figsize=(12, 8))
        x_loadings = best_model.x_loadings_
        x_vars = plsr_results["X_vars"]

        # Limit to top 20 variables by absolute loading value for readability
        if len(x_vars) > 20:
            # Get indices of top 20 variables by absolute loading value
            top_indices = np.argsort(np.abs(x_loadings[:, 0]))[-20:]
            x_loadings = x_loadings[top_indices, :]
            x_vars = [x_vars[i] for i in top_indices]

        plt.barh(range(len(x_vars)), x_loadings[:, 0], align='center')
        plt.yticks(range(len(x_vars)), x_vars)
        plt.xlabel('Component 1 Loading')
        plt.title(f'PLSR X Loadings (Component 1) - Best Model from Fold {best_model_idx + 1}')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "plsr_cv_best_model_x_loadings.png"), dpi=300)

        # Plot Y loadings for the best model from cross-validation
        plt.figure(figsize=(12, 8))
        y_loadings = best_model.y_loadings_
        y_vars = plsr_results["Y_vars"]

        plt.barh(range(len(y_vars)), y_loadings[:, 0], align='center')
        plt.yticks(range(len(y_vars)), y_vars)
        plt.xlabel('Component 1 Loading')
        plt.title(f'PLSR Y Loadings (Component 1) - Best Model from Fold {best_model_idx + 1}')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "plsr_cv_best_model_y_loadings.png"), dpi=300)

    # Visualize the final model trained on the entire dataset
    if 'final_model' in plsr_results:
        final_model = plsr_results['final_model']
        final_best_n_comp = plsr_results['final_best_n_components']

        # Plot X loadings for the final model
        plt.figure(figsize=(12, 8))
        x_loadings = final_model.x_loadings_
        x_vars = plsr_results["X_vars"]

        # Limit to top 20 variables by absolute loading value for readability
        if len(x_vars) > 20:
            # Get indices of top 20 variables by absolute loading value
            top_indices = np.argsort(np.abs(x_loadings[:, 0]))[-20:]
            x_loadings = x_loadings[top_indices, :]
            x_vars = [x_vars[i] for i in top_indices]

        plt.barh(range(len(x_vars)), x_loadings[:, 0], align='center')
        plt.yticks(range(len(x_vars)), x_vars)
        plt.xlabel('Component 1 Loading')
        plt.title(f'PLSR X Loadings (Component 1) - Final Model with {final_best_n_comp} Components')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "plsr_cv_final_model_x_loadings.png"), dpi=300)

        # Plot Y loadings for the final model
        plt.figure(figsize=(12, 8))
        y_loadings = final_model.y_loadings_
        y_vars = plsr_results["Y_vars"]

        plt.barh(range(len(y_vars)), y_loadings[:, 0], align='center')
        plt.yticks(range(len(y_vars)), y_vars)
        plt.xlabel('Component 1 Loading')
        plt.title(f'PLSR Y Loadings (Component 1) - Final Model with {final_best_n_comp} Components')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "plsr_cv_final_model_y_loadings.png"), dpi=300)

        # If there are multiple components, visualize the explained variance
        if final_best_n_comp > 1:
            # Calculate explained variance for the final model
            X = data_df[plsr_results["X_vars"]].values
            Y = data_df[plsr_results["Y_vars"]].values

            if args.scale:
                X = StandardScaler().fit_transform(X)
                Y = StandardScaler().fit_transform(Y)

            X_scores = final_model.transform(X)
            Y_scores = final_model.y_scores_

            x_explained_variance = np.var(X_scores, axis=0) / np.var(X, axis=0).sum()
            y_explained_variance = np.var(Y_scores, axis=0) / np.var(Y, axis=0).sum()

            plt.figure(figsize=(10, 6))
            components = range(1, final_best_n_comp + 1)
            plt.bar(components, x_explained_variance, alpha=0.7, label='X Variance')
            plt.bar(components, y_explained_variance, alpha=0.7, label='Y Variance')
            plt.xlabel('Component')
            plt.ylabel('Explained Variance Ratio')
            plt.title(f'PLSR Explained Variance by Component - Final Model with {final_best_n_comp} Components')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "plsr_cv_final_model_explained_variance.png"), dpi=300)

    print(f"PLSR visualizations saved to {args.output_dir}")
    print("PLSR analysis complete!")


def run_snf_analysis(args):
    """Run the SNF analysis pipeline."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading imputed NHANES data from {args.data_path}...")
    data_df = pd.read_csv(args.data_path)

    # Define variables for analysis
    cognitive_vars = ["CFDRIGHT"]  # Cognitive function right responses
    covariates = ["RIDAGEYR", "female", "male", "black", "mexican", "other_hispanic", "other_eth", "SES_LEVEL",
                  "education", "SDDSRVYR"]  # Demographics and survey cycle

    # Since we no longer have description data, we'll create exposure categories based on column name patterns
    print("Creating exposure categories based on column name patterns...")

    # Define patterns for different exposure categories
    exposure_patterns = {
        "heavy metals": ["LBX", "PB", "CD", "HG", "SE", "MN"],
        "pesticides": ["LBX", "DDE", "DDT", "HCH", "NAL", "PAR", "CYF", "DME"],
        "phenols": ["LBX", "BP", "PH", "TR", "BP3", "BPA"],
        "phthalates": ["LBX", "PHT", "MHP", "MBP", "MZP", "MOP", "MCH", "MOH"],
        "nutrients": ["LBX", "FOL", "B12", "VIC", "VID", "VIA", "VIE"],
        "cotinine": ["LBX", "COT"],
        "pcbs": ["LBX", "PCB"],
        "dioxins": ["LBX", "DIO", "PCD", "HXC"],
        "furans": ["LBX", "FUR", "PCP", "HXC"],
        "hydrocarbons": ["LBX", "PAH", "FLU", "PHE", "PYR"],
        "perchlorate": ["LBX", "PER", "SCN", "NIT"],
        "phytoestrogens": ["LBX", "EQU", "DAI", "GEN"],
        "polybrominated ethers": ["LBX", "PBD", "BDE"],
        "polyflourochemicals": ["LBX", "PFO", "PFH", "PFN", "PFD"],
        "volatile compounds": ["LBX", "VOC", "BEN", "TOL", "XYL"]
    }

    # Create a dictionary mapping exposure categories to variable names
    exposure_categories = {}

    # Assign variables to categories based on patterns
    for category, patterns in exposure_patterns.items():
        category_vars = []
        for col in data_df.columns:
            # Check if any pattern matches the column name
            if any(pattern in col for pattern in patterns):
                category_vars.append(col)

        # Only add the category if it has variables
        if category_vars:
            exposure_categories[category] = category_vars

    print(f"Identified {len(exposure_categories)} exposure categories")
    for category, vars_list in exposure_categories.items():
        print(f"  {category}: {len(vars_list)} variables")

    print(f"Running SNF with {len(exposure_categories)} exposure categories, {len(cognitive_vars)} cognitive variables, and {len(covariates)} covariates...")

    # Run SNF
    from excog_trajectory import analysis
    snf_results = analysis.run_snf(
        data=data_df,
        exposure_categories=exposure_categories,
        cognitive_vars=cognitive_vars,
        covariates=covariates,
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
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans

    # Plot the fused similarity matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(snf_results["fused_matrix"], cmap='viridis')
    plt.colorbar(label='Similarity')
    plt.title('SNF Fused Similarity Matrix')
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
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=cluster_labels, cmap='viridis', alpha=0.8)
    plt.colorbar(scatter, label='Cluster')
    plt.title('t-SNE Visualization of SNF Fused Similarity Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "snf_tsne.png"), dpi=300)

    print(f"SNF visualizations saved to {args.output_dir}")
    print("SNF analysis complete!")


def main():
    """Main entry point for the CLI."""
    args = parse_args()

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
    else:
        print(f"Unknown command: {args.command}")
        exit(1)


if __name__ == "__main__":
    main()
