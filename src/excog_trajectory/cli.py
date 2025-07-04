#!/usr/bin/env python3
"""
Command-line interface for the excog_trajectory package.

This module provides CLI commands for analyzing exposomic trajectories
of cognitive decline in NHANES data and for downloading NHANES data.
"""

import argparse
import os

import pandas as pd

from excog_trajectory import analysis, data, visualization


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
        default="data/processed/combined/imputed_nhanes_dat1.csv",
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
        default=5,
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
        nhanes_data[dat] = data.apply_qc_rules(nhanes_data[dat],
                                               cognitive_vars,
                                               covariates=covariates,
                                               standardize=True,
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
        # Exclude original covariates and dummy variables created from categorical covariates
        categorical_covariates = ["Cycle", "RIAGENDR", "RIDRETH1"]
        dummy_prefixes = [f"{cov}_" for cov in categorical_covariates]

        # Filter out covariates and any column that starts with dummy variable prefixes
        exposure_vars = [col for col in nhanes_data[dat].columns 
                        if col not in cognitive_vars + covariates and 
                        not any(col.startswith(prefix) for prefix in dummy_prefixes)]

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

    # Check if we have the original 'Cycle' column or dummy variables
    cycle_dummy_cols = [col for col in combined_data.columns if col.startswith('Cycle_')]

    if 'Cycle' in combined_data.columns:
        # Original Cycle column exists, use it for grouping
        print("Using original Cycle column for filtering...")
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
    elif cycle_dummy_cols:
        # Cycle has been converted to dummy variables, use them for grouping
        print(f"Using Cycle dummy variables for filtering: {cycle_dummy_cols}")
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
        # No Cycle information available, keep all columns
        print("Warning: No Cycle column or dummy variables found. Keeping all columns.")
        columns_to_keep = combined_data.columns.tolist()

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
    # Exclude original covariates and dummy variables created from categorical covariates
    categorical_covariates = ["Cycle", "RIAGENDR", "RIDRETH1"]
    dummy_prefixes = [f"{cov}_" for cov in categorical_covariates]

    # Filter out covariates and any column that starts with dummy variable prefixes
    exposure_vars = [col for col in combined_data.columns 
                    if col not in cognitive_vars + covariates and 
                    not any(col.startswith(prefix) for prefix in dummy_prefixes)]
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
            data=imputed_data.drop(columns=["sample", "Cycle"]),
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
    cognitive_vars = ["CFDDS"]  # Cognitive function right responses
    covariates = ["Cycle", "RIDAGEYR", "RIAGENDR", "INDFMPIR",
                  "DMDEDUC2", "RIDRETH1"]  # Demographics and survey cycle
    # Identify exposure variables (all variables that are not cognitive or covariates)
    all_vars = set(data_df.columns)
    non_exposure_vars = set(cognitive_vars + covariates)
    exposure_vars = list(all_vars - non_exposure_vars)
    x = data_df[exposure_vars + covariates].drop(columns=["Cycle"])
    y = data_df[cognitive_vars]

    print(f"Running PLSR with {len(exposure_vars)} exposure variables, {len(cognitive_vars)} cognitive variables, and {len(covariates)} covariates...")

    if args.n_repetitions > 1:
        print(
            f"Running PLSR with double cross-validation ({args.outer_folds} outer folds, {args.inner_folds} inner folds) repeated {args.n_repetitions} times...")
    else:
        print(
            f"Running PLSR with double cross-validation ({args.outer_folds} outer folds, {args.inner_folds} inner folds)...")

    plsr_results = analysis.pls_double_cv(
        x=x,
        y=y,
        n_repeats=args.n_repetitions,
        max_components=args.max_components,
    )

    # Print information about the final model
    mode = int(plsr_results['table']['LV'].mode()[0])
    print(f"\nThe most repeated number of LV: {str(mode)}")
    from sklearn.cross_decomposition import PLSRegression
    best_model = PLSRegression(
        n_components=mode, scale=True, max_iter=1000).fit(
        X=x, y=y
    )
    print(f"A final model has been trained on the entire dataset using {str(mode)} components.")

    # Save the results
    import pickle
    with open(os.path.join(args.output_dir, "best_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)

    print(f"PLSR results saved to {os.path.join(args.output_dir, 'best_model.pkl')}")
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
