import unittest
import pandas as pd
import numpy as np
from excog_trajectory.analysis import run_wgcna


class TestWGCNA(unittest.TestCase):
    def setUp(self):
        # Create a simple test dataset with correlated variables
        np.random.seed(42)
        n_samples = 100
        n_features = 20

        # Create groups of correlated variables
        group1 = np.random.normal(0, 1, (n_samples, 5))
        group2 = np.random.normal(0, 1, (n_samples, 5))
        group3 = np.random.normal(0, 1, (n_samples, 5))

        # Add some noise to create correlations within groups
        data = np.zeros((n_samples, n_features))

        # Group 1: variables 0-4
        for i in range(5):
            data[:, i] = group1[:, 0] + 0.2 * group1[:, i] + 0.1 * np.random.normal(0, 1, n_samples)

        # Group 2: variables 5-9
        for i in range(5):
            data[:, i+5] = group2[:, 0] + 0.2 * group2[:, i] + 0.1 * np.random.normal(0, 1, n_samples)

        # Group 3: variables 10-14
        for i in range(5):
            data[:, i+10] = group3[:, 0] + 0.2 * group3[:, i] + 0.1 * np.random.normal(0, 1, n_samples)

        # Covariates: variables 15-17
        data[:, 15] = np.random.normal(0, 1, n_samples)  # age
        data[:, 16] = np.random.binomial(1, 0.5, n_samples)  # gender
        data[:, 17] = np.random.normal(0, 1, n_samples)  # education

        # Cognitive variables: variables 18-19
        data[:, 18] = np.random.normal(0, 1, n_samples)  # cognitive test 1
        data[:, 19] = 0.7 * data[:, 18] + 0.3 * np.random.normal(0, 1, n_samples)  # cognitive test 2

        # Create column names
        exposure_vars = [f"exposure_{i}" for i in range(15)]
        covariate_vars = ["age", "gender", "education"]
        cognitive_vars = ["cognitive_1", "cognitive_2"]

        # Create DataFrame
        self.df = pd.DataFrame(
            data, 
            columns=exposure_vars + covariate_vars + cognitive_vars
        )

        # Define variable groups for testing
        self.exposure_vars = exposure_vars
        self.covariate_vars = covariate_vars
        self.cognitive_vars = cognitive_vars

        # Create a dataset with missing values
        df_with_missing = self.df.copy()

        # Add missing values to specific variables
        # Group 1: 10% missing
        for i in range(3):
            mask = np.random.choice([True, False], size=n_samples, p=[0.1, 0.9])
            df_with_missing.loc[mask, f"exposure_{i}"] = np.nan

        # Group 2: 20% missing
        for i in range(5, 8):
            mask = np.random.choice([True, False], size=n_samples, p=[0.2, 0.8])
            df_with_missing.loc[mask, f"exposure_{i}"] = np.nan

        # Group 3: 30% missing
        for i in range(10, 13):
            mask = np.random.choice([True, False], size=n_samples, p=[0.3, 0.7])
            df_with_missing.loc[mask, f"exposure_{i}"] = np.nan

        # Add one variable with excessive missing data (80%)
        mask = np.random.choice([True, False], size=n_samples, p=[0.8, 0.2])
        df_with_missing.loc[mask, "exposure_14"] = np.nan

        self.df_with_missing = df_with_missing

    def test_wgcna_basic(self):
        """Test that WGCNA runs without errors and returns expected structure."""
        results = run_wgcna(self.df)

        # Check that results contain expected keys
        self.assertIn('clusters', results)
        self.assertIn('adjacency', results)
        self.assertIn('dendrogram', results)
        self.assertIn('labels', results)
        self.assertIn('filtered_variables', results)
        self.assertIn('missing_assessment', results)

        # Check that clusters is a dictionary
        self.assertIsInstance(results['clusters'], dict)

        # Check that labels DataFrame has expected columns
        self.assertIn('variable', results['labels'].columns)
        self.assertIn('cluster', results['labels'].columns)

    def test_wgcna_exclude_covariates(self):
        """Test that WGCNA correctly excludes covariates."""
        results = run_wgcna(self.df, covariates=self.covariate_vars)

        # Check that no covariate variables are in the labels
        for var in self.covariate_vars:
            self.assertNotIn(var, results['labels']['variable'].values)

    def test_wgcna_exclude_cognitive(self):
        """Test that WGCNA correctly excludes cognitive variables."""
        results = run_wgcna(self.df, cognitive_vars=self.cognitive_vars)

        # Check that no cognitive variables are in the labels
        for var in self.cognitive_vars:
            self.assertNotIn(var, results['labels']['variable'].values)

    def test_wgcna_clusters(self):
        """Test that WGCNA identifies meaningful clusters."""
        results = run_wgcna(
            self.df, 
            covariates=self.covariate_vars,
            cognitive_vars=self.cognitive_vars,
            power=6,
            min_module_size=3,
            cut_height=0.5
        )

        # Check that we have at least one cluster
        self.assertGreater(len(results['clusters']), 0)

        # Check that each cluster has at least min_module_size variables
        for cluster_id, variables in results['clusters'].items():
            self.assertGreaterEqual(len(variables), 3)

    def test_wgcna_with_missing_data(self):
        """Test that WGCNA handles missing data correctly."""
        results = run_wgcna(
            self.df_with_missing,
            covariates=self.covariate_vars,
            cognitive_vars=self.cognitive_vars,
            min_observations=10,
            min_pct_observations=0.1,
            min_reliable_correlations_pct=0.3
        )

        # Check that results contain expected keys for missing data assessment
        self.assertIn('missing_assessment', results)
        self.assertIn('filtered_variables', results)

        # Check that missing_assessment contains expected keys
        missing_assessment = results['missing_assessment']
        self.assertIn('missing_info', missing_assessment)
        self.assertIn('pairwise_complete_count', missing_assessment)
        self.assertIn('pairwise_complete_pct', missing_assessment)
        self.assertIn('pairwise_reliability', missing_assessment)
        self.assertIn('reliability_info', missing_assessment)

        # Check that filtered_variables contains expected keys
        filtered_info = results['filtered_variables']
        self.assertIn('filtered_variables', filtered_info)
        self.assertIn('filtered_reasons', filtered_info)
        self.assertIn('kept_variables', filtered_info)

        # Check that the variable with excessive missing data was filtered out
        self.assertIn('exposure_14', filtered_info['filtered_variables'])

        # Check that we still have clusters despite missing data
        self.assertGreater(len(results['clusters']), 0)

    def test_wgcna_critical_variables(self):
        """Test that critical variables are always included regardless of missing data."""
        # Define a critical variable that would normally be filtered out
        critical_vars = ['exposure_14']  # This has 80% missing data

        results = run_wgcna(
            self.df_with_missing,
            covariates=self.covariate_vars,
            cognitive_vars=self.cognitive_vars,
            min_observations=10,
            min_pct_observations=0.1,
            min_reliable_correlations_pct=0.3,
            critical_variables=critical_vars
        )

        # Check that the critical variable is included in the kept variables
        self.assertIn('exposure_14', results['filtered_variables']['kept_variables'])

        # Check that the critical variable appears in the labels
        self.assertIn('exposure_14', results['labels']['variable'].values)

    def test_wgcna_without_missing_assessment(self):
        """Test that WGCNA works when missing data assessment is disabled."""
        results = run_wgcna(
            self.df_with_missing,
            covariates=self.covariate_vars,
            cognitive_vars=self.cognitive_vars,
            assess_missing=False
        )

        # Check that missing_assessment is not in the results
        self.assertNotIn('missing_assessment', results)

        # Check that all variables are included (no filtering)
        for var in self.exposure_vars:
            self.assertIn(var, results['labels']['variable'].values)


if __name__ == '__main__':
    unittest.main()
