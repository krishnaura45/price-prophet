import logging
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# For Log transform
def log_transform(df: pd.DataFrame, features: list) -> pd.DataFrame:
    df = df.copy()
    for feature in features:
        df[feature] = np.log1p(df[feature])
    return df

# For adding new features
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Bath_total"] = df["Full Bath"] + 0.5 * df["Half Bath"] + df["Bsmt Full Bath"] + 0.5 * df["Bsmt Half Bath"]
    df["BsmtFinSF"] = df["BsmtFin SF 1"] + df["BsmtFin SF 2"] - df["Bsmt Unf SF"]
    df["Porch"] = df["Open Porch SF"] + df["Enclosed Porch"] + df["3Ssn Porch"] + df["Screen Porch"]
    df["FinSF"] = df["1st Flr SF"] + df["2nd Flr SF"] - df["Low Qual Fin SF"]
    return df

# For dropping redundant/low-impact base features after engineering new features from them
def drop_redundant_features(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = ['Pool QC', 'Pool Area', 'Misc Feature', 'Misc Val', 'Alley', 'Fence', 'Wood Deck SF',
                    'Fireplace Qu', 'Fireplaces', 'Mas Vnr Type', 'Mas Vnr Area',
                    'Full Bath', 'Half Bath', 'Bsmt Full Bath', 'Bsmt Half Bath',
                    'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Open Porch SF',
                    'Enclosed Porch', '3Ssn Porch', 'Screen Porch',
                    '1st Flr SF', '2nd Flr SF', 'Low Qual Fin SF']
    return df.drop(columns=cols_to_drop, errors='ignore')

def fill_remaining_nas(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_none = ['Bsmt Qual','Bsmt Cond','Bsmt Exposure','BsmtFin Type 1','BsmtFin Type 2',
                    'Garage Type','Garage Finish','Garage Qual','Garage Cond']
    cols_to_zero = ['Garage Yr Blt', 'Garage Area', 'Garage Cars']
    df[cols_to_none] = df[cols_to_none].fillna("None")
    df[cols_to_zero] = df[cols_to_zero].fillna(0)
    return df

def impute_lot_frontage(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop_cols = ["Lot Frontage", "SalePrice", "Id"]
    train = df[df["Lot Frontage"].notna()]
    test = df[df["Lot Frontage"].isna()]
    
    features = train.select_dtypes(exclude="object").drop(columns=drop_cols, errors='ignore')
    target = train["Lot Frontage"]

    reg = XGBRegressor(n_estimators=5000, learning_rate=0.01)
    reg.fit(features, target)

    test_feat = test[features.columns]
    df.loc[test.index, "Lot Frontage"] = reg.predict(test_feat)
    return df

def finalize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_cols = ['Year Built', 'Year Remod/Add', 'Garage Yr Blt']
    df[date_cols] = (df[date_cols] / 5).round() * 5
    df["Year Built"].replace({1885.0: 1890.0, 1870.0: 1880.0, 1875.0: 1880.0}, inplace=True)
    df["Garage Yr Blt"].replace({1905.0: 1910.0, 2205.0: 2005.0}, inplace=True)

    cat_cols = ['MS SubClass', 'Year Built', 'Year Remod/Add', 'Garage Yr Blt', 'Yr Sold', 'Mo Sold']
    df[cat_cols] = df[cat_cols].astype(str)
    return df

def clean_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['MS Zoning'].replace(['A (agr)', 'I (all)'], 'C (all)', inplace=True)
    df['Neighborhood'].replace(['Greens', 'GrnHill', 'Landmrk'], 'NAmes', inplace=True)
    df['House Style'].replace({'2.5Fin': '2.5Unf'}, inplace=True)
    df['Exterior 1st'].replace(['Stone', 'PreCast', 'ImStucc'], 'BrkComm', inplace=True)
    df['Exterior 2nd'].replace(['PreCast', 'Other'], 'CBlock', inplace=True)
    df['Bsmt Qual'].replace('Po', 'Fa', inplace=True)
    df['Kitchen Qual'].replace('Po', 'Fa', inplace=True)
    df['Sale Type'].replace({'VWD': 'Con', 'WD ': 'WD'}, inplace=True)
    return df

def handle_condition(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Condition'] = df['Condition 1'] + '_' + df['Condition 2']
    df.drop(['Condition 1', 'Condition 2'], axis=1, inplace=True)

    condition_counts = df['Condition'].value_counts()
    rare_conditions = condition_counts[condition_counts < 5].index
    df['Condition'] = df['Condition'].replace(rare_conditions, 'Other')
    df['Condition'] = df['Condition'].replace(['RRAn_Feedr', 'RRNn_Norm', 'RRNe_Norm', 'PosN_PosN', 'Feedr_Feedr'], 'Other')
    return df

def ordinal_encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5, 'No': 1, 'Mn': 2, 'Av': 3,
               'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6,
               'RFn': 2, 'Fin': 3, 'N': 1, 'P': 2, 'Y': 3, 'MnWw': 1, 'GdWo': 2,
               'MnPrv': 3, 'GdPrv': 4, 'Other': 1}
    cols_OE = ['Exter Qual', 'Exter Cond', 'Bsmt Qual', 'Bsmt Exposure', 'BsmtFin Type 1',
               'BsmtFin Type 2', 'Heating QC', 'Kitchen Qual', 'Garage Finish']
    df[cols_OE] = df[cols_OE].replace(mapping)
    return df

def drop_weak_features(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop_2 = ['Street', 'Land Contour', 'Utilities', 'Land Slope', 'Roof Matl', 'Bsmt Cond',
                      'Heating', 'Central Air', 'Electrical', 'Functional', 'Garage Qual', 
                      'Garage Cond', 'Paved Drive']
    return df.drop(cols_to_drop_2, axis=1, errors='ignore')

def heavy_feature_engineering(df: pd.DataFrame, features: list) -> pd.DataFrame:
    df = log_transform(df, features)
    df = add_derived_features(df)
    df = drop_redundant_features(df)
    df = fill_remaining_nas(df)
    df = impute_lot_frontage(df)
    df = finalize_categoricals(df)
    df = clean_categories(df)
    df = handle_condition(df)
    df = ordinal_encode(df)
    df = drop_weak_features(df)
    return df