# import json
# import numpy as np
# import pandas as pd
# import requests
# from zenml import step

# @step(enable_cache=False)
# def predictor(
#     service,  # unused placeholder to satisfy pipeline
#     input_data: str,
# ) -> np.ndarray:
#     data = json.loads(input_data)
#     data.pop("columns", None)
#     data.pop("index", None)

#     expected_columns = [
#         "Order", "PID", "MS SubClass", "Lot Frontage", "Lot Area", "Overall Qual",
#         "Overall Cond", "Year Built", "Year Remod/Add", "Mas Vnr Area", "BsmtFin SF 1",
#         "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF", "1st Flr SF", "2nd Flr SF",
#         "Low Qual Fin SF", "Gr Liv Area", "Bsmt Full Bath", "Bsmt Half Bath",
#         "Full Bath", "Half Bath", "Bedroom AbvGr", "Kitchen AbvGr", "TotRms AbvGrd",
#         "Fireplaces", "Garage Yr Blt", "Garage Cars", "Garage Area", "Wood Deck SF",
#         "Open Porch SF", "Enclosed Porch", "3Ssn Porch", "Screen Porch", "Pool Area",
#         "Misc Val", "Mo Sold", "Yr Sold"
#     ]

#     df = pd.DataFrame(data["data"], columns=expected_columns)
#     json_list = json.loads(json.dumps(list(df.T.to_dict().values())))

#     response = requests.post(
#         url="http://127.0.0.1:1234/invocations",
#         json={"inputs": json_list},
#         headers={"Content-Type": "application/json"},
#     )

#     if response.status_code != 200:
#         raise Exception(f"Prediction failed: {response.text}")

#     return np.array(response.json())

import json
import numpy as np
import pandas as pd
import requests
from zenml import step

@step(enable_cache=False)
def predictor(
    service,  # unused placeholder to satisfy pipeline
    input_data: str,
) -> np.ndarray:
    data = json.loads(input_data)
    data.pop("columns", None)
    data.pop("index", None)

    expected_columns = [
        "MS SubClass", "MS Zoning", "Lot Frontage", "Lot Area", "Lot Shape", "Lot Config", "Neighborhood", 
        "Bldg Type", "House Style", "Overall Qual", "Overall Cond", "Year Built", "Year Remod/Add", 
        "Roof Style", "Exterior 1st", "Exterior 2nd", "Exter Qual", "Exter Cond", "Foundation", "Bsmt Qual", 
        "Bsmt Exposure", "BsmtFin Type 1", "BsmtFin Type 2", "Total Bsmt SF", "Heating QC", "Gr Liv Area", 
        "Bedroom AbvGr", "Kitchen AbvGr", "Kitchen Qual", "TotRms AbvGrd", "Garage Type", "Garage Yr Blt", 
        "Garage Finish", "Garage Cars", "Garage Area", "Mo Sold", "Yr Sold", "Sale Type", "Sale Condition", 
        "Id", "Bath_total", "BsmtFinSF", "Porch", "FinSF", "Condition"
    ]

    df = pd.DataFrame(data["data"], columns=expected_columns)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))

    response = requests.post(
        url="http://127.0.0.1:1234/invocations",
        json={"inputs": json_list},
        headers={"Content-Type": "application/json"},
    )

    if response.status_code != 200:
        raise Exception(f"Prediction failed: {response.text}")

    return np.expm1(np.array(response.json()))