import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from PIL import Image

# MLflow model REST endpoint (ensure manual MLflow model server is running)
MLFLOW_URL = "http://127.0.0.1:1234/invocations"

st.set_page_config(page_title="Price Prophet", layout="wide")
st.title("🏠 House Price Predictor (USA)")

whole_pipeline_image = Image.open("assets/basic-overview.png")

st.markdown(
    """ 
#### What can I do? 
    Predict the sale price for a given house in America based on features like overall quality, above ground living area, year built, etc. I will be using the following deployed pipeline to predict house price.  """
)
st.image(whole_pipeline_image, caption="Whole Pipeline")

st.sidebar.header("📋 Data Input Options")
input_mode = st.sidebar.radio("Choose input mode", ["Manual Entry", "Upload CSV"])

# Final features model expects (post-preprocessing)
FEATURES = [
    "Overall Qual", "Gr Liv Area", "Garage Cars", "Total Bsmt SF", "Garage Area", "FinSF", "Year Built", "Bath_total", "MS SubClass", "MS Zoning", "Lot Frontage", "Lot Area", "Lot Shape", "Lot Config", "Neighborhood", "Bldg Type", "House Style", "Overall Cond", "Year Remod/Add",
    "Roof Style", "Exterior 1st", "Exterior 2nd", "Exter Qual", "Exter Cond", "Foundation", "Bsmt Qual","Bsmt Exposure", "BsmtFin Type 1", "BsmtFin Type 2", "Heating QC", "Bedroom AbvGr", "Kitchen AbvGr", "Kitchen Qual", "TotRms AbvGrd", "Garage Type", "Garage Yr Blt", "Garage Finish", "Mo Sold", "Yr Sold", "Sale Type", "Sale Condition", "Id", "BsmtFinSF", "Porch", "Condition"
]

# Ordinal encoding map
ORDINAL_MAP = {
    'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5,
    'No': 1, 'Mn': 2, 'Av': 3,
    'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6,
    'RFn': 2, 'Fin': 3,
    'N': 1, 'P': 2, 'Y': 3,
    'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4,
    'Other': 1
}
ORDINAL_COLS = [
    'Exter Qual', 'Exter Cond', 'Bsmt Qual', 'Bsmt Exposure', 'BsmtFin Type 1',
    'BsmtFin Type 2', 'Heating QC', 'Kitchen Qual', 'Garage Finish'
]

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[ORDINAL_COLS] = df[ORDINAL_COLS].replace(ORDINAL_MAP)
    return df

def predict(input_df: pd.DataFrame):
    df_encoded = preprocess(input_df)
    inputs = df_encoded.to_dict(orient="records")
    response = requests.post(
        MLFLOW_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps({"inputs": inputs}),
    )
    if response.status_code == 200:
        preds = response.json()["predictions"]
        return [round(float(p), 2) for p in map(np.expm1, preds)]
    else:
        st.error(f"❌ MLflow error {response.status_code}: {response.text}")
        return None

if input_mode == "Manual Entry":
    st.subheader("🔧 Enter Feature Details")

    input_dict = {}
    cols = st.columns(3)
    for i, feature in enumerate(FEATURES):
        with cols[i % 3]:
            if feature == "MS SubClass":
                input_dict[feature] = st.selectbox(
                    feature, [20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190], index=0
                )
            elif feature == "MS Zoning":
                input_dict[feature] = st.selectbox(
                    feature, ['C (all)', 'FV', 'RH', 'RL', 'RM'], index=3
                )
            elif feature == "Lot Shape":
                input_dict[feature] = st.selectbox(
                    feature, ['IR1', 'IR2', 'IR3', 'Reg'], index=3
                )
            elif feature == "Lot Config":
                input_dict[feature] = st.selectbox(
                    feature, ['Corner', 'CulDSac', 'FR2', 'FR3', 'Inside'], index=4
                )
            elif feature == "Neighborhood":
                input_dict[feature] = st.selectbox(
                    feature, ['NAmes', 'Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr','Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel', 'NPkVill', 'NWAmes', 'NoRidge', 'NridgHt', 'OldTown','SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber','Veenker'], index=0
                )
            elif feature == "Bldg Type":
                input_dict[feature] = st.selectbox(
                    feature, ['1Fam', '2fmCon', 'Duplex', 'Twnhs', 'TwnhsE'], index=0
                )
            elif feature == "House Style":
                input_dict[feature] = st.selectbox(
                    feature, ['1.5Fin', '1.5Unf', '1Story', '2.5Unf', '2Story', 'SFoyer', 'SLvl'], index=2
                )
            elif feature == "Condition":
                input_dict[feature] = st.selectbox(
                    feature, ['Artery_Norm', 'Feedr_Norm', 'Norm_Norm', 'Other', 'PosA_Norm','PosN_Norm', 'RRAe_Norm', 'RRAn_Norm'], index=2
                )
            elif feature == "Roof Style":
                input_dict[feature] = st.selectbox(
                    feature, ['Flat', 'Gable', 'Gambrel', 'Hip', 'Mansard', 'Shed'], index=1
                )
            elif feature == "Exterior 1st":
                input_dict[feature] = st.selectbox(
                    feature, ['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'MetalSd', 'Plywood', 'Stucco', 'VinylSd', 'Wd Sdng','WdShing'], index=10
                )
            elif feature == "Exterior 2nd":
                input_dict[feature] = st.selectbox(
                    feature, ['AsbShng', 'AsphShn', 'Brk Cmn', 'BrkFace', 'CBlock', 'CmentBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Plywood', 'Stone', 'Stucco', 'VinylSd', 'Wd Sdng', 'Wd Shng'], index=12
                )
            elif feature == "Foundation":
                input_dict[feature] = st.selectbox(
                    feature, ['BrkTil', 'CBlock', 'PConc', 'Slab', 'Stone', 'Wood'], index=1
                )
            elif feature == "Garage Type":
                input_dict[feature] = st.selectbox(
                    feature, ['2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd','None'], index=1
                )
            elif feature == "Sale Type":
                input_dict[feature] = st.selectbox(
                    feature, ['COD', 'CWD', 'Con', 'ConLD', 'ConLI', 'ConLw', 'New', 'Oth', 'WD'], index=8
                )
            elif feature == "Sale Condition":
                input_dict[feature] = st.selectbox(
                    feature, ['Abnorml', 'AdjLand', 'Alloca', 'Family', 'Normal', 'Partial'], index=4
                )
            elif feature in ["Year Built", "Year Remod/Add", "Garage Yr Blt"]:
                input_dict[feature] = st.text_input(feature, value=1960)
            elif feature == "Yr Sold":
                input_dict[feature] = st.text_input(feature, value=2010)


            elif feature == "Overall Qual":
                input_dict[feature] = st.selectbox(feature, list(range(1, 11)), index=4)
            elif feature == "Overall Cond":
                input_dict[feature] = st.selectbox(feature, list(range(1, 11)), index=6)
            elif feature == "Mo Sold":
                months = list(range(1, 13))
                input_dict[feature] = st.selectbox(feature, months, index=4)

            elif feature in ["Lot Frontage", "Total Bsmt SF", "Garage Cars", "Garage Area", "Bath_total", "BsmtFinSF"]:
                input_dict[feature] = st.number_input(feature, value=0.0)
            else:
                input_dict[feature] = st.number_input(feature, value=0)

    if st.button("Predict"):
        input_df = pd.DataFrame([input_dict])
        prediction = predict(input_df)
        if prediction:
            st.success(f"💰 Predicted Sale Price: ${prediction[0]:,.2f}")

elif input_mode == "Upload CSV":
    st.subheader("📤 Upload CSV File")
    file = st.file_uploader("Upload file with correct feature columns", type=["csv"])
    if file:
        input_df = pd.read_csv(file)
        st.dataframe(input_df)

        if st.button("Predict"):
            predictions = predict(input_df)
            if predictions:
                input_df["Predicted SalePrice"] = predictions
                st.dataframe(input_df)
                sub = input_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Predictions", sub, "predictions.csv", "text/csv")