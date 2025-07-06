import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from zenml import step

from src.model_evaluator import ModelEvaluator, RegressionModelEvaluationStrategy

@step
def model_evaluation_step(
    trained_model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[dict, float]:
    """
    Evaluates the trained model using ModelEvaluator and RegressionModelEvaluationStrategy.

    Parameters:
    trained_model (RegressorMixin): The trained model.
    X_test (pd.DataFrame): The test data features.
    y_test (pd.Series): The test data target.

    Returns:
    Tuple[dict, float]: Evaluation metrics and MSE value.
    """
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a pandas Series.")

    logging.info("Starting model evaluation step.")

    logging.info(f"✅ Model input schema: {X_test.columns.tolist()}")

    evaluator = ModelEvaluator(strategy=RegressionModelEvaluationStrategy())
    evaluation_metrics = evaluator.evaluate(trained_model, X_test, y_test)

    if not isinstance(evaluation_metrics, dict):
        raise ValueError("Evaluation metrics must be returned as a dictionary.")

    logging.info("Evaluation step completed successfully.")
    return evaluation_metrics, evaluation_metrics.get("Mean Squared Error")