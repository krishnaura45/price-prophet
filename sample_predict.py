import json
import requests

# MLflow model server URL (ensure it's running on this port)
url = "http://127.0.0.1:1234/invocations"

# Single input record — replace with real feature values
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
        "Yr Sold": 2010
    }
]

# Proper MLflow REST input format
json_data = json.dumps({"inputs": input_data})
headers = {"Content-Type": "application/json"}

# POST to MLflow
response = requests.post(url, headers=headers, data=json_data)

if response.status_code == 200:
    print("✅ Prediction:", response.json())
else:
    print(f"❌ Error {response.status_code}: {response.text}")