import logging
from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, BaseEstimator, clone, TransformerMixin
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        pass


class CatBoostTunedStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        best_params = {
            'learning_rate': 0.05853851006330062,
            'depth': 7,
            'l2_leaf_reg': 3.245808021961012,
            'bagging_temperature': 0.6935813036577569,
            'random_strength': 0.7724852356271209,
            'border_count': 151
        }
        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_indices = [X_train.columns.get_loc(c) for c in cat_cols]

        model = CatBoostRegressor(
            **best_params,
            iterations=1000,
            loss_function='RMSE',
            cat_features=cat_indices,
            verbose=100,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model


class IndividualModelStrategy(ModelBuildingStrategy):
    def __init__(self, model: RegressorMixin):
        self.model = model

    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
        num_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="mean"))
        ])
        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols)
        ])

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", self.model)
        ])

        pipeline.fit(X_train, y_train)
        return pipeline


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.models_ = [clone(m) for m in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)


class AveragedEnsembleStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        model_1 = XGBRegressor(
            colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05,
            max_depth=3, min_child_weight=1.7817, n_estimators=2200,
            reg_alpha=0.4640, reg_lambda=0.8571, subsample=0.5213,
            silent=1, random_state=7, nthread=-1)

        model_2 = LGBMRegressor(
            objective='regression', num_leaves=5, learning_rate=0.05,
            n_estimators=720, max_bin=55, bagging_fraction=0.8,
            bagging_freq=5, feature_fraction=0.2319,
            feature_fraction_seed=9, bagging_seed=9,
            min_data_in_leaf=6, min_sum_hessian_in_leaf=11)

        combo_model = AveragingModels(models=(model_1, model_2))
        return IndividualModelStrategy(combo_model).build_and_train_model(X_train, y_train)


class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy):
        logging.info("Switching model building strategy.")
        self._strategy = strategy

    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        logging.info("Building and training the model using the selected strategy.")
        return self._strategy.build_and_train_model(X_train, y_train)


if __name__ == "__main__":
    pass