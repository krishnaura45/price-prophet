from typing import Tuple
import pandas as pd
from src.data_splitter import DataSplitter, SimpleTrainTestSplitStrategy
from zenml import step

@step
def data_splitting_step(
    df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the data into training and testing sets using DataSplitter and a chosen strategy.
    Also logs the shape of each split for traceability.
    """
    splitter = DataSplitter(strategy=SimpleTrainTestSplitStrategy(test_size, random_state))
    X_train, X_test, y_train, y_test = splitter.split(df, target_column)

    print(f"\n📊 Data split summary:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test