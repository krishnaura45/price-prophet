import json
import requests
import pandas as pd

url = "http://127.0.0.1:1234/invocations"

input_data = [
    {
        "Order": 1,
        "PID": 5286,
        "MS SubClass": 20,
        "Lot Frontage": 80.0,
        "Lot Area": 9600,
        "Overall Qual": 5,
        "Overall Cond": 7,
        "Year Built": 1961,
        "Year Remod/Add": 1961,
        "Mas Vnr Area": 0.0,
        "BsmtFin SF 1": 700.0,
        "BsmtFin SF 2": 0.0,
        "Bsmt Unf SF": 150.0,
        "Total Bsmt SF": 850.0,
        "1st Flr SF": 856,
        "2nd Flr SF": 854,
        "Low Qual Fin SF": 0,
        "Gr Liv Area": 1710.0,
        "Bsmt Full Bath": 1,
        "Bsmt Half Bath": 0,
        "Full Bath": 1,
        "Half Bath": 0,
        "Bedroom AbvGr": 3,
        "Kitchen AbvGr": 1,
        "TotRms AbvGrd": 7,
        "Fireplaces": 2,
        "Garage Yr Blt": 1961,
        "Garage Cars": 2,
        "Garage Area": 500.0,
        "Wood Deck SF": 210.0,
        "Open Porch SF": 0,
        "Enclosed Porch": 0,
        "3Ssn Porch": 0,
        "Screen Porch": 0,
        "Pool Area": 0,
        "Misc Val": 0,
        "Mo Sold": 5,
        "Yr Sold": 2010,
    },
    {
        "Order": 2,
        "PID": 1234,
        "MS SubClass": 60,
        "Lot Frontage": 70.0,
        "Lot Area": 8500,
        "Overall Qual": 9,
        "Overall Cond": 5,
        "Year Built": 1995,
        "Year Remod/Add": 1996,
        "Mas Vnr Area": 150.0,
        "BsmtFin SF 1": 600.0,
        "BsmtFin SF 2": 100.0,
        "Bsmt Unf SF": 100.0,
        "Total Bsmt SF": 800.0,
        "1st Flr SF": 900,
        "2nd Flr SF": 850,
        "Low Qual Fin SF": 0,
        "Gr Liv Area": 1350.0,
        "Bsmt Full Bath": 1,
        "Bsmt Half Bath": 0,
        "Full Bath": 2,
        "Half Bath": 1,
        "Bedroom AbvGr": 4,
        "Kitchen AbvGr": 1,
        "TotRms AbvGrd": 8,
        "Fireplaces": 1,
        "Garage Yr Blt": 1995,
        "Garage Cars": 2,
        "Garage Area": 480.0,
        "Wood Deck SF": 120.0,
        "Open Porch SF": 30,
        "Enclosed Porch": 0,
        "3Ssn Porch": 0,
        "Screen Porch": 0,
        "Pool Area": 0,
        "Misc Val": 0,
        "Mo Sold": 6,
        "Yr Sold": 2011,
    },
]

headers = {"Content-Type": "application/json"}
json_data = json.dumps({"inputs": input_data})

response = requests.post(url, headers=headers, data=json_data)

if response.status_code == 200:
    result = response.json()
    predictions = result["predictions"]

    df = pd.DataFrame(input_data)
    df["Predicted Sale Price"] = predictions
    df.to_csv("predictions.csv", index=False)
    print("✅ Saved predictions to predictions.csv")
else:
    print(f"❌ Error {response.status_code}: {response.text}")