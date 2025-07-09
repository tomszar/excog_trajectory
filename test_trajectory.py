import pickle

import pandas as pd

from excog_trajectory import analysis, columns, data, visualization, trajectory

model_path = "results/plsr/best_model.pkl"
data_path = "data/processed/imputed/imputed_nhanes_dat1.csv"

# Load data and set it up
with open(model_path, "rb") as f:
    model = pickle.load(f)
df = pd.read_csv(data_path)
x_scores = pd.DataFrame(model.x_scores_)
x_scores.columns = [f"LV{i+1}" for i in range(x_scores.shape[1])]
df_full = pd.concat([df, x_scores], axis=1)
df_full.set_index("sample", inplace=True)

# New
covariates = columns.COVARIATES
valid_covariates = columns.validate_columns(df, covariates, raise_error=False)
covariates_cat = columns.CATEGORICAL_COVARIATES
cognitive_cat = columns.COGNITIVE_CAT
dummy_vars = columns.get_dummy_vars(df_full, covariates_cat)

initial_model_matrix = df[cognitive_cat + dummy_vars + valid_covariates]

# Test the new create_proper_model_matrix function
proper_model_matrix = trajectory.create_model_matrix(
    model_matrix=initial_model_matrix,
    cognitive_cat=cognitive_cat,
    dummy_vars=dummy_vars,
    valid_covariates=valid_covariates,
    add_interactions=True
)
reduced_model = trajectory.create_model_matrix(
    model_matrix=initial_model_matrix,
    cognitive_cat=cognitive_cat,
    dummy_vars=dummy_vars,
    valid_covariates=valid_covariates,
    add_interactions=False
)

y = x_scores
betas = trajectory.estimate_betas(proper_model_matrix, y)
ls_vectors = trajectory._get_ls_vectors(proper_model_matrix)
contrast = [[0,1,2], [3,4,5]]

deltas, angles, shapes = trajectory.estimate_difference(y,
                                                        proper_model_matrix,
                                                        ls_vectors,
                                                        contrast)

r_deltas, r_angles, r_shapes = trajectory.RRPP(y,
                                               proper_model_matrix,
                                               reduced_model,
                                               ls_vectors,
                                               contrast,
                                               9999)