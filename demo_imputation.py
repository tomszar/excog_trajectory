#!/usr/bin/env python3
"""
Script to demonstrate the imputation procedure with random variable selection,
saving multiple imputed datasets with appropriate names.
"""

import os
import argparse
from excog_trajectory import data

def main():
    """Run the imputation with random variable selection and save multiple imputed datasets."""
    # Create output directories if they don't exist
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Run the imputation with 10 randomly selected variables
    print("Running imputation with 10 randomly selected variables...")
    kernel = data.impute_exposure_variables(
        n_random_vars=10,
        random_state=42,
        n_iterations=3,  # Run 3 iterations as specified
        n_imputations=5,  # Generate 5 datasets as specified
    )

if __name__ == "__main__":
    main()
