#!/usr/bin/env python3
"""
Script to demonstrate the imputation procedure with random variable selection,
saving both the imputed dataset and performance metrics to files.
"""

import os
import argparse
from excog_trajectory import data

def main():
    """Run the imputation with random variable selection and save both the imputed dataset and performance metrics."""
    # Create output directories if they don't exist
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Run the imputation with 10 randomly selected variables
    print("Running imputation with 10 randomly selected variables...")
    imputed_data, performance_metrics = data.impute_exposure_variables(
        data_path="data/processed/cleaned_nhanes.csv",
        output_path="data/processed/imputed_nhanes_10vars.csv",
        cv_results_path="results/cv_results.csv",
        plot_path="results/imputation_methods_comparison.png",
        n_random_vars=10,
        random_state=42,
        n_jobs=14,
    )

    # Print summary of imputation performance
    print("\nImputation Performance Summary:")
    print("-------------------------------")

    # The performance metrics now have a different structure
    # Each variable has metrics for multiple imputation methods
    import numpy as np

    # Create a list to store all metrics
    all_metrics = []

    # Process the metrics for each variable and method
    for var, methods in performance_metrics.items():
        # Handle both old and new structures of performance_metrics
        if isinstance(methods, dict) and any(isinstance(v, dict) for v in methods.values()):
            # New structure: each variable has metrics for multiple imputation methods
            for method_name, metrics in methods.items():
                all_metrics.append({
                    "variable": var,
                    "method": method_name,
                    "rmse": metrics["rmse"],
                    "rmse_std": metrics.get("rmse_std", 0),
                    "mae": metrics.get("mae", 0)
                })
        else:
            # Old structure: each variable has a single set of metrics
            # Assume default method is "RandomForest"
            all_metrics.append({
                "variable": var,
                "method": "RandomForest",
                "rmse": methods.get("rmse", 0) if isinstance(methods, dict) else 0,
                "rmse_std": methods.get("rmse_std", 0) if isinstance(methods, dict) else 0,
                "mae": methods.get("mae", 0) if isinstance(methods, dict) else 0
            })

    # Convert to DataFrame
    import pandas as pd
    metrics_df = pd.DataFrame(all_metrics)

    # Calculate average metrics by method
    method_avg = metrics_df.groupby('method').agg({
        'rmse': 'mean',
        'rmse_std': 'mean',
        'mae': 'mean'
    }).reset_index()

    # Print average metrics by method
    print("\nAverage metrics by imputation method:")
    for _, row in method_avg.iterrows():
        print(f"{row['method']}: RMSE={row['rmse']:.4f}±{row['rmse_std']:.4f}, MAE={row['mae']:.4f}")

    # Find the best method (lowest average RMSE)
    best_method = method_avg.loc[method_avg['rmse'].idxmin(), 'method']
    print(f"\nBest imputation method: {best_method}")

    # Add a 'method_type' column to identify averages
    method_avg['variable'] = 'AVERAGE'

    # Combine the detailed metrics with the averages
    metrics_df = pd.concat([metrics_df, method_avg])

    # Save to CSV
    metrics_file = os.path.join("results", "imputation_performance.csv")
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Performance metrics saved to {metrics_file}")

    print("\nImputation complete!")

if __name__ == "__main__":
    main()
