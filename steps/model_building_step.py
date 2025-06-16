import logging
from typing import Annotated

import mlflow
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from zenml import ArtifactConfig, step
from zenml.client import Client

# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker
from zenml import Model

model = Model(
    name="prices_predictor_2.0",
    version=None,
    license="Apache 2.0",
    description="Price prediction model for houses in USA.",
)

# Simple stacking strategy - averaging base models
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    # do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)

lin_reg = LinearRegression()
lasso = Lasso(alpha =0.0005, random_state=1)
ENet = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Annotated[Pipeline, ArtifactConfig(name="sklearn_pipeline", is_model_artifact=True)]:
    """
    Builds and trains a simple stacked model wrapped in a pipeline.

    Parameters:
    X_train (pd.DataFrame): The training data features.
    y_train (pd.Series): The training data labels/target.

    Returns:
    Pipeline: The trained scikit-learn pipeline including preprocessing and the stacked model.
    """
    # Ensure the inputs are of the correct type
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    # Identify categorical and numerical columns
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
    num_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

    logging.info(f"Categorical columns: {cat_cols.tolist()}")
    logging.info(f"Numerical columns: {num_cols.tolist()}")

    # Define preprocessing for categorical and numerical features
    # numerical_transformer = SimpleImputer(strategy="mean")
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    # # Define the model training pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", AveragingModels(models = (model_lgb, model_xgb, lin_reg)))])

    # Start an MLflow run to log the model training process
    if not mlflow.active_run():
        mlflow.start_run()  # Start a new MLflow run if there isn't one active

    try:
        # Enable autologging for scikit-learn to automatically capture model metrics, parameters, and artifacts
        mlflow.sklearn.autolog()

        logging.info("Building and training the Linear Regression model.")
        pipeline.fit(X_train, y_train)
        logging.info("Model training completed.")

        # Log the columns that the model expects
        onehot_encoder = (
            pipeline.named_steps["preprocessor"].transformers_[1][1].named_steps["onehot"]
        )
        onehot_encoder.fit(X_train[cat_cols])
        expected_columns = num_cols.tolist() + list(
            onehot_encoder.get_feature_names_out(cat_cols)
        )
        logging.info(f"Model expects the following columns: {expected_columns}")

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e

    finally:
        # End the MLflow run
        mlflow.end_run()

    return pipeline
