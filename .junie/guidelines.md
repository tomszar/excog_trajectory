# Project Guidelines for Junie

## Project Overview

This project, "Exposomic Trajectories of Cognitive Decline in NHANES," analyzes the relationships between environmental exposures and cognitive decline using data from the National Health and Nutrition Examination Survey (NHANES). The goal is to investigate how various environmental exposures (such as metals, pesticides, and other chemicals) may influence cognitive trajectories over time in the US population.

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
│       ├── visualization.py # Visualization functions
│       └── cli.py           # Command-line interface
├── tests/
│   └── unit/        # Unit tests
├── pyproject.toml   # Project configuration
└── README.md        # Project documentation
```

## Key Components

1. **Data Module (`data.py`)**: 
   - Functions for loading, preprocessing, and cleaning NHANES data
   - Quality control rules implementation
   - Missing value imputation using Multiple Imputation by Chained Equations (MICE)
   - Data downloading and extraction utilities

2. **Analysis Module (`analysis.py`)**: 
   - Statistical methods for analyzing relationships between exposures and cognitive outcomes
   - Linear regression models
   - Longitudinal analysis for assessing exposure effects on cognitive trajectories
   - Mixture modeling to identify patterns of exposures

3. **CLI Module (`cli.py`)**: 
   - Command-line interface for the package
   - Commands for downloading data, running analysis, and imputing missing values

## Testing Guidelines

When making changes to the codebase, Junie should:

1. Run the appropriate unit tests to ensure the changes don't break existing functionality:
   ```bash
   pytest tests/unit/
   ```

2. For changes to data processing functions, verify that the quality control rules are still properly applied.

3. For changes to analysis functions, ensure that the statistical methods are correctly implemented and produce valid results.

## Code Style Guidelines

This project follows specific code style guidelines:

1. Uses Black for code formatting with a line length of 88 characters
2. Uses isort for import sorting with the Black profile
3. Uses mypy for type checking with strict settings
4. Uses ruff for linting with E, F, B, and I rule sets

When making changes, Junie should:

1. Format code with Black:
   ```bash
   black src/ tests/
   ```

2. Sort imports with isort:
   ```bash
   isort src/ tests/
   ```

3. Check types with mypy:
   ```bash
   mypy src/
   ```

4. Lint code with ruff:
   ```bash
   ruff check src/ tests/
   ```

## Dependencies

The project requires Python 3.10 or higher and depends on several scientific computing libraries:
- numpy (≥1.20.0)
- pandas (2.3.0)
- scipy (≥1.7.0)
- matplotlib (≥3.4.0)
- seaborn (≥0.11.0)
- scikit-learn (1.7.0)
- statsmodels (≥0.13.0)

Development dependencies include:
- pytest (≥7.0.0)
- pytest-cov (≥3.0.0)
- black (≥22.0.0)
- isort (≥5.10.0)
- mypy (≥0.950)
- ruff (≥0.0.100)

## Building and Installation

The project uses hatchling as its build system. To build the project:

```bash
pip install build
python -m build
```

To install the project in development mode:

```bash
uv pip install -e .
```

For development dependencies:

```bash
uv pip install -e ".[dev]"
```
