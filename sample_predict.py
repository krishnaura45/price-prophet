import json
import requests
import numpy as np

# Endpoint where MLflow model is being served
url = "http://127.0.0.1:1234/invocations"

# Input record aligned with expected_columns.json
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
        "Condition": "Norm_Norm",
    }
]

# Format and send request
headers = {"Content-Type": "application/json"}
payload = json.dumps({"inputs": input_data})
response = requests.post(url, headers=headers, data=payload)

# Interpret response
if response.status_code == 200:
    log_prediction = response.json()["predictions"][0]
    actual_prediction = np.expm1(log_prediction)
    print(f"✅ Predicted SalePrice: ${actual_prediction:,.2f}")
else:
    print(f"❌ Error {response.status_code}: {response.text}")
