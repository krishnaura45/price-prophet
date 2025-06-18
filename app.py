import streamlit as st
import pandas as pd
import requests
import json
from PIL import Image

# MLflow model REST endpoint
MLFLOW_URL = "http://127.0.0.1:1234/invocations"

st.set_page_config(page_title="Price Prophet", layout="wide")
st.title("🏠 House Price Predictor (USA)")

whole_pipeline_image = Image.open("assets/whole_pipeline.png")

st.markdown(
    """ 
#### What can I do? 
    The objective here is to predict the sale price for a given house in America based on features like overall quality, above ground living area, year built, etc. I will be using the following deployed pipeline to predict house price.  """
)
st.image(whole_pipeline_image, caption="Whole Pipeline")

st.sidebar.header("📋 Data Input Options")
input_mode = st.sidebar.radio("Choose input mode", ["Manual Entry", "Upload CSV"])

# Feature list (same as model expects)
FEATURES = [
    "Order", "PID", "MS SubClass", "Lot Frontage", "Lot Area",
    "Overall Qual", "Overall Cond", "Year Built", "Year Remod/Add",
    "Mas Vnr Area", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF",
    "Total Bsmt SF", "1st Flr SF", "2nd Flr SF", "Low Qual Fin SF",
    "Gr Liv Area", "Bsmt Full Bath", "Bsmt Half Bath", "Full Bath",
    "Half Bath", "Bedroom AbvGr", "Kitchen AbvGr", "TotRms AbvGrd",
    "Fireplaces", "Garage Yr Blt", "Garage Cars", "Garage Area",
    "Wood Deck SF", "Open Porch SF", "Enclosed Porch", "3Ssn Porch",
    "Screen Porch", "Pool Area", "Misc Val", "Mo Sold", "Yr Sold",
]

def predict(input_df: pd.DataFrame):
    """Send POST request to MLflow and return predictions"""
    inputs = input_df.to_dict(orient="records")
    response = requests.post(
        MLFLOW_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps({"inputs": inputs}),
    )
    if response.status_code == 200:
        return response.json()["predictions"]
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
                    feature, [20, 30, 40, 50, 60, 70, 80, 85, 90, 120, 160, 180, 190], index=0
                )
            elif feature == "Overall Qual":
                input_dict[feature] = st.selectbox(feature, list(range(1, 11)), index=5)
            elif feature == "Overall Cond":
                input_dict[feature] = st.selectbox(feature, list(range(1, 10)), index=5)
            elif feature == "Mo Sold":
                months = list(range(1, 13))
                input_dict[feature] = st.selectbox(feature, months, index=4)
            elif feature == "Yr Sold":
                input_dict[feature] = st.selectbox(feature, list(range(2006, 2011)), index=4)
            else:
                input_dict[feature] = st.number_input(feature, value=0.0)

    if st.button("Predict"):
        input_df = pd.DataFrame([input_dict])
        prediction = predict(input_df)
        if prediction:
            st.success(f"💰 Predicted Sale Price: ${prediction[0]:,.3f}K")

elif input_mode == "Upload CSV":
    st.subheader("📤 Upload CSV File")
    file = st.file_uploader("Upload file with correct feature columns", type=["csv"])
    if file:
        input_df = pd.read_csv(file)
        st.dataframe(input_df)

        if st.button("Predict"):
            predictions = predict(input_df)
            if predictions:
                input_df["SalePrice (1000 $)"] = predictions
                st.dataframe(input_df)
                sub = input_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Predictions", sub, "predictions.csv", "text/csv")