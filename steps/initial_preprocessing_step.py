import pandas as pd
from src.data_preprocessing import (
    missing_value_analyzer,
    partial_mv_handler
)
from zenml import step


@step
def initial_preprocessing_step(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """Performs initial preprocessing: drops identifiers, analyzes and fills missing values."""

    # Log and analyze missing data
    missing_report = missing_value_analyzer(df)
    print("\n🔍 Missing Value Report (Top 5):\n", missing_report.head(5))

    # Apply partial missing value handler
    cleaned_df = partial_mv_handler(df, strategy=strategy)

    # Final null check
    remaining_missing = cleaned_df.isnull().sum()
    remaining_missing = remaining_missing[remaining_missing > 0]
    print("\n❗ Remaining Missing Values:\n", remaining_missing)

    return cleaned_df