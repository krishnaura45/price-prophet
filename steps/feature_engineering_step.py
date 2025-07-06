import pandas as pd
from src.feature_engineering import heavy_feature_engineering
from zenml import step

@step
def feature_engineering_step(
    df: pd.DataFrame, features: list = None
) -> pd.DataFrame:
    """Applies full heavy feature engineering pipeline to the input dataframe."""

    # Ensuring that features is a list, even if not provided
    if features is None:
        features = []  # or raise an error if features are required

    transformed_df = heavy_feature_engineering(df, features)
    return transformed_df