import logging
import pandas as pd
from sklearn.impute import SimpleImputer

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def missing_value_analyzer(df: pd.DataFrame) -> pd.DataFrame:
    """Analyzes and summarizes missing values in the DataFrame."""
    miss_df = pd.DataFrame(index=df.columns)
    miss_df['zero_count'] = (df == 0).sum()
    miss_df['nan_count'] = df.isna().sum()
    miss_df['none_str_count'] = (df == "None").sum()
    miss_df['total_missing'] = miss_df.sum(axis=1)
    miss_df['missing_percent'] = 100 * miss_df['total_missing'] / len(df)
    miss_df['dtype'] = df.dtypes
    return miss_df.sort_values(by='missing_percent', ascending=False)


def partial_mv_handler(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """Handles partial missing values for selected categorical and numerical features."""
    cleaned_df = df.copy()

    # Drop identifiers and duplicates
    cleaned_df.drop(['Order', 'PID'], axis=1, inplace=True, errors='ignore')
    cleaned_df.drop_duplicates(inplace=True)
    cleaned_df.reset_index(drop=True, inplace=True)
    cleaned_df["Id"] = range(len(cleaned_df))

    # Define columns with missing data
    cat_feat = ['Garage Finish', 'Garage Qual', 'Garage Cond', 'MS Zoning',
                'Utilities', 'Functional', 'Exterior 2nd', 'Exterior 1st', 'Sale Type', 'Kitchen Qual']
    num_feat = ['Bsmt Full Bath', 'Bsmt Half Bath', 'Garage Cars', 'Garage Area',
                'Total Bsmt SF', 'Bsmt Unf SF', 'BsmtFin SF 2', 'BsmtFin SF 1']

    # Impute Missing Values
    if strategy == "mean":
        cat_imputer = SimpleImputer(strategy='most_frequent')
        cleaned_df[cat_feat] = cat_imputer.fit_transform(cleaned_df[cat_feat])

        num_imputer = SimpleImputer(strategy='mean')
        cleaned_df[num_feat] = num_imputer.fit_transform(cleaned_df[num_feat])
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    return cleaned_df


if __name__ == "__main__":
    pass