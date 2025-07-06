# import logging
# from typing import Annotated

# import mlflow
# import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.base import RegressorMixin
# from sklearn.preprocessing import OneHotEncoder, RobustScaler
# from zenml import ArtifactConfig, step
# from zenml.client import Client
# from zenml import Model

# from src.model_building import (
#     ModelBuilder,
#     CatBoostTunedStrategy,
#     IndividualModelStrategy,
#     AveragingModels,
#     AveragedEnsembleStrategy,
# )

# # Get the active experiment tracker from ZenML
# experiment_tracker = Client().active_stack.experiment_tracker

# model = Model(
#     name="prices_predictor_optimum",
#     version=None,
#     license="Apache 2.0",
#     description="Price prediction model for houses.",
# )

# @step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
# def model_building_step(
#     X_train: pd.DataFrame, y_train: pd.Series, strategy: str = "catboost_tuned"
# ) -> Annotated[Pipeline, ArtifactConfig(name="sklearn_pipeline", is_model_artifact=True)]:
#     """
#     Builds and trains a model using the selected strategy and returns a trained pipeline.

#     Parameters:
#     X_train (pd.DataFrame): Training data features.
#     y_train (pd.Series): Training labels.
#     strategy (str): Model strategy identifier.

#     Returns:
#     Pipeline: Trained scikit-learn pipeline with preprocessing and model.
#     """
#     if not isinstance(X_train, pd.DataFrame):
#         raise TypeError("X_train must be a pandas DataFrame.")
#     if not isinstance(y_train, pd.Series):
#         raise TypeError("y_train must be a pandas Series.")

#     logging.info(f"Using model building strategy: {strategy}")

#     if strategy == "catboost_tuned":
#         builder = ModelBuilder(CatBoostTunedStrategy())
#     elif strategy == "individual":
#         builder = ModelBuilder(IndividualModelStrategy())
#     elif strategy == "avg_xgb_lgbm":
#         builder = ModelBuilder(AveragedEnsembleStrategy())
#     else:
#         raise ValueError(f"Unsupported model strategy: {strategy}")

#     if not mlflow.active_run():
#         mlflow.start_run()

#     try:
#         mlflow.sklearn.autolog()
#         model_pipeline = builder.build_model(X_train, y_train)
#         logging.info("Model training completed.")
#     except Exception as e:
#         logging.error(f"Model training failed: {e}")
#         raise e
#     finally:
#         mlflow.end_run()

#     return model_pipeline

import logging
from typing import Annotated
import json
import mlflow
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from zenml import ArtifactConfig, step
from zenml.client import Client
from zenml import Model

# Dummy transformer to wrap CatBoost in sklearn pipeline
class IdentityTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker

model = Model(
    name="prices_predictor_optimum",
    version=None,
    license="Apache 2.0",
    description="Price prediction model for houses."
)

@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Annotated[Pipeline, ArtifactConfig(name="sklearn_pipeline", is_model_artifact=True)]:
    """
    Builds and trains a CatBoost model wrapped in a dummy sklearn pipeline.

    Parameters:
    X_train (pd.DataFrame): The training data features.
    y_train (pd.Series): The training data labels/target.

    Returns:
    Pipeline: A dummy sklearn pipeline wrapping CatBoost model.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import numpy as np

    # Identify categorical columns and convert
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    X_train[categorical_cols] = X_train[categorical_cols].astype(str)

    cat_feature_indices = [X_train.columns.get_loc(col) for col in categorical_cols]

    best_params = {
        'learning_rate': 0.05853851006330062,
        'depth': 7,
        'l2_leaf_reg': 3.245808021961012,
        'bagging_temperature': 0.6935813036577569,
        'random_strength': 0.7724852356271209,
        'border_count': 151
    }

    cb_model = CatBoostRegressor(
        **best_params,
        iterations=1000,
        loss_function='RMSE',
        cat_features=cat_feature_indices,
        verbose=100,
        random_state=42
    )

    # Wrap CatBoost in a dummy pipeline to satisfy ZenML's expected Pipeline output
    pipeline = Pipeline([
        ("identity", IdentityTransformer()),
        ("catboost", cb_model)
    ])

    if not mlflow.active_run():
        mlflow.start_run()

    try:
        mlflow.sklearn.autolog()
        logging.info("Training CatBoost pipeline model.")
        pipeline.fit(X_train, y_train)
        logging.info("Model training completed.")

        # # Uncomment block to save expected column names after training
        # expected_columns = X_train.columns.tolist()
        # with open("expected_columns.json", "w") as f:
        #     json.dump(expected_columns, f)

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e

    finally:
        mlflow.end_run()

    return pipeline