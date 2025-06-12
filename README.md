# Exposomic Trajectories of Cognitive Decline in NHANES

This project analyzes the relationships between environmental exposures and cognitive decline using data from the National Health and Nutrition Examination Survey (NHANES).

## Project Overview

The goal of this project is to investigate how various environmental exposures (such as metals, pesticides, and other chemicals) may influence cognitive trajectories over time in the US population, using NHANES data as a representative sample.

## Features

- Load and preprocess NHANES data related to cognitive assessments and environmental exposures
- Apply quality control rules to ensure data reliability:
  - Remove variables with less than 200 non-NaN values
  - Remove categorical variables with less than 200 values in a category
  - Remove variables with 90% of non-NaN values equal to zero
- Impute missing values in exposure variables using Multiple Imputation by Chained Equations (MICE):
  - Consider variable types (continuous vs categorical) for appropriate imputation methods
  - Use cross-validation to assess imputation performance
  - Focus imputation only on exposure variables
- Analyze relationships between exposures and cognitive outcomes using various statistical methods
- Examine longitudinal trajectories of cognitive decline in relation to exposure levels
- Visualize exposure-outcome relationships and analysis results

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone the repository
git clone https://github.com/yourusername/excog-trajectory.git
cd excog-trajectory

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies using uv
uv pip install -e .

# For development dependencies
uv pip install -e ".[dev]"
```

## Usage

### Command Line Interface

The package provides a command-line interface (CLI) with the following commands:

#### Download NHANES Data

```bash
# Download NHANES data to the data/raw directory
excog download

# Specify a different output directory
excog download --output-dir path/to/directory

# Specify a different DOI or filename
excog download --doi 10.5061/dryad.example --filename custom_name.zip

# Use a direct URL instead of the Data Dryad API
excog download --direct-url https://example.com/nhanes_data.zip
```

#### Run Analysis

```bash
# Run the analysis pipeline with default settings
excog analyze

# Specify a different NHANES cycle
excog analyze --cycle "2013-2014"

# Specify a different data path or output directory
excog analyze --data-path path/to/data --output-dir path/to/results
```

#### Impute Missing Values

```bash
# Run the imputation procedure with default settings
excog impute

# Specify a different data path or output path
excog impute --data-path path/to/cleaned_data.csv --output-path path/to/imputed_data.csv

# Specify a variable description file
excog impute --description-path path/to/description.csv

# Customize imputation parameters
excog impute --n-imputations 10 --n-cv-folds 10 --random-state 123
```

### Python API

You can also use the package as a Python library:

```python
from excog_trajectory import data, analysis, visualization

# Load NHANES data
nhanes_data = data.load_nhanes_data(cycle="2011-2012")

# Extract cognitive and exposure data
cognitive_data = data.get_cognitive_data(nhanes_data)
exposure_data = data.get_exposure_data(nhanes_data)

# Merge the data
merged_data = data.merge_cognitive_exposure_data(cognitive_data, exposure_data)

# Impute missing values in exposure variables
imputed_data, performance_metrics = data.impute_exposure_variables(
    data_path="data/processed/cleaned_nhanes.csv",
    output_path="data/processed/imputed_nhanes.csv",
    n_cv_folds=5,
    random_state=42
)

# Print imputation performance metrics
for var, metrics in performance_metrics.items():
    print(f"{var}: RMSE = {metrics['rmse']:.4f}, MAE = {metrics['mae']:.4f}")

# Run analysis on the imputed data
model_results = analysis.run_linear_models(
    data=imputed_data,
    outcome_vars=["CFDDS", "CFDST"],  # Digit Symbol Substitution Test scores
    exposure_vars=["LBXBPB", "LBXBCD"],  # Blood lead and cadmium levels
    covariates=["RIDAGEYR", "RIAGENDR", "RIDRETH1", "DMDEDUC2"]  # Demographics
)

# Visualize results
fig = visualization.plot_model_coefficients(model_results, ["LBXBPB", "LBXBCD"])
fig.savefig("results/model_coefficients.png")
```

## Project Structure

```
excog-trajectory/
├── data/
│   ├── raw/         # Raw NHANES data files
│   └── processed/   # Processed data files
├── src/
│   └── excog_trajectory/
│       ├── __init__.py
│       ├── data.py          # Data loading and processing functions
│       ├── analysis.py      # Statistical analysis functions
│       └── visualization.py # Visualization functions
├── tests/
│   └── unit/        # Unit tests
├── pyproject.toml   # Project configuration
└── README.md        # Project documentation
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
