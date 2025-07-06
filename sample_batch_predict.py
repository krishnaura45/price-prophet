import json
import requests
import pandas as pd
import numpy as np

url = "http://127.0.0.1:1234/invocations"

# Aligned with expected_columns.json
input_data = [
    {
        "MS SubClass": "20",
        "MS Zoning": "RL",
        "Lot Frontage": 80.0,
        "Lot Area": 9600,
        "Lot Shape": "Reg",
        "Lot Config": "Inside",
        "Neighborhood": "NAmes",
        "Bldg Type": "1Fam",
        "House Style": "1Story",
        "Overall Qual": 5,
        "Overall Cond": 7,
        "Year Built": "1960",
        "Year Remod/Add": "1960",
        "Roof Style": "Gable",
        "Exterior 1st": "VinylSd",
        "Exterior 2nd": "VinylSd",
        "Exter Qual": 3,
        "Exter Cond": 3,
        "Foundation": "CBlock",
        "Bsmt Qual": 3,
        "Bsmt Exposure": 3,
        "BsmtFin Type 1": 6,
        "BsmtFin Type 2": 0,
        "Total Bsmt SF": 850.0,
        "Heating QC": 4,
        "Gr Liv Area": 1710.0,
        "Bedroom AbvGr": 3,
        "Kitchen AbvGr": 1,
        "Kitchen Qual": 3,
        "TotRms AbvGrd": 7,
        "Garage Type": "Attchd",
        "Garage Yr Blt": "1960",
        "Garage Finish": 3,
        "Garage Cars": 2,
        "Garage Area": 500.0,
        "Mo Sold": "5",
        "Yr Sold": "2010",
        "Sale Type": "WD",
        "Sale Condition": "Normal",
        "Id": 0,
        "Bath_total": 1.5,
        "BsmtFinSF": 700.0,
        "Porch": 0.0,
        "FinSF": 1710.0,
        "Condition": "Norm_Norm"
    },
    {
        "MS SubClass": "60",
        "MS Zoning": "RL",
        "Lot Frontage": 70.0,
        "Lot Area": 8500,
        "Lot Shape": "IR1",
        "Lot Config": "Corner",
        "Neighborhood": "CollgCr",
        "Bldg Type": "1Fam",
        "House Style": "2Story",
        "Overall Qual": 9,
        "Overall Cond": 5,
        "Year Built": "1995",
        "Year Remod/Add": "1996",
        "Roof Style": "Hip",
        "Exterior 1st": "VinylSd",
        "Exterior 2nd": "VinylSd",
        "Exter Qual": 5,
        "Exter Cond": 3,
        "Foundation": "PConc",
        "Bsmt Qual": 4,
        "Bsmt Exposure": 3,
        "BsmtFin Type 1": 6,
        "BsmtFin Type 2": 0,
        "Total Bsmt SF": 800.0,
        "Heating QC": 5,
        "Gr Liv Area": 1350.0,
        "Bedroom AbvGr": 4,
        "Kitchen AbvGr": 1,
        "Kitchen Qual": 5,
        "TotRms AbvGrd": 8,
        "Garage Type": "Attchd",
        "Garage Yr Blt": "1995",
        "Garage Finish": 3,
        "Garage Cars": 2,
        "Garage Area": 480.0,
        "Mo Sold": "6",
        "Yr Sold": "2011",
        "Sale Type": "WD",
        "Sale Condition": "Normal",
        "Id": 1,
        "Bath_total": 2.5,
        "BsmtFinSF": 700.0,
        "Porch": 30.0,
        "FinSF": 1750.0,
        "Condition": "Norm_PosN"
    }
]

headers = {"Content-Type": "application/json"}
response = requests.post(url, headers=headers, data=json.dumps({"inputs": input_data}))

if response.status_code == 200:
    log_preds = response.json()["predictions"]
    saleprice_preds = [np.expm1(p) for p in log_preds]

    df = pd.DataFrame(input_data)
    df["Predicted SalePrice"] = saleprice_preds
    df.to_csv("predictions.csv", index=False)
    print("✅ Saved predictions to predictions.csv")
else:
    print(f"❌ Error {response.status_code}: {response.text}")