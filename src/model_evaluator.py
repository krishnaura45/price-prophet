import logging
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Model Evaluation Strategy
class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        pass


# Concrete Strategy for Regression Model Evaluation
class RegressionModelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        logging.info("Predicting using the trained model.")
        y_pred = model.predict(X_test)

        logging.info("Calculating evaluation metrics.")
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            "Mean Squared Error": mse,
            "RMSE": rmse,
            "R-Squared": r2,
        }

        logging.info(f"Model Evaluation Metrics: {metrics}")
        return metrics


# Optional: Separate CatBoost scoring logic (same as regression)
def evaluate_catboost_model(
    model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    logging.info("Evaluating CatBoost model explicitly.")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    metrics = {"Mean Squared Error": mse, "RMSE": rmse, "R-Squared": r2}
    logging.info(f"CatBoost Evaluation Metrics: {metrics}")
    return metrics


# Context Class for Model Evaluation
class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluationStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ModelEvaluationStrategy):
        logging.info("Switching model evaluation strategy.")
        self._strategy = strategy

    def evaluate(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        logging.info("Evaluating the model using the selected strategy.")
        return self._strategy.evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    pass