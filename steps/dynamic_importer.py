import pandas as pd
from zenml import step
import numpy as np
import json

@step
def dynamic_importer() -> str:
    """Dynamically generates test data for prediction with the CatBoost model."""
    data = {
        "MS SubClass": [20, 60],
        "MS Zoning": ["RL", "RM"],
        "Lot Frontage": [80.0, 70.0],
        "Lot Area": [9600, 8500],
        "Lot Shape": ["Reg", "IR1"],
        "Lot Config": ["Inside", "Corner"],
        "Neighborhood": ["NAmes", "CollgCr"],
        "Bldg Type": ["1Fam", "1Fam"],
        "House Style": ["2Story", "1Story"],
        "Overall Qual": [5, 9],
        "Overall Cond": [7, 5],
        "Year Built": [1961, 1995],
        "Year Remod/Add": [1961, 1996],
        "Roof Style": ["Gable", "Hip"],
        "Exterior 1st": ["VinylSd", "HdBoard"],
        "Exterior 2nd": ["VinylSd", "HdBoard"],
        "Exter Qual": ["TA", "Gd"],
        "Exter Cond": ["TA", "TA"],
        "Foundation": ["CBlock", "PConc"],
        "Bsmt Qual": ["Gd", "Gd"],
        "Bsmt Exposure": ["No", "Mn"],
        "BsmtFin Type 1": ["GLQ", "ALQ"],
        "BsmtFin Type 2": ["Unf", "Unf"],
        "Total Bsmt SF": [850, 800],
        "Heating QC": ["Ex", "Gd"],
        "Gr Liv Area": [1710.0, 1350.0],
        "Bedroom AbvGr": [3, 4],
        "Kitchen AbvGr": [1, 1],
        "Kitchen Qual": ["TA", "Gd"],
        "TotRms AbvGrd": [7, 8],
        "Garage Type": ["Attchd", "Attchd"],
        "Garage Yr Blt": [1961, 1995],
        "Garage Finish": ["RFn", "Fin"],
        "Garage Cars": [2, 2],
        "Garage Area": [500.0, 480.0],
        "Mo Sold": [5, 6],
        "Yr Sold": [2010, 2011],
        "Sale Type": ["WD", "WD"],
        "Sale Condition": ["Normal", "Normal"],
        "Id": [0, 1],
        "Bath_total": [2.0, 3.0],
        "BsmtFinSF": [700.0, 600.0],
        "Porch": [0.0, 30.0],
        "FinSF": [1710.0, 1750.0],
        # "LotFrontage": [80.0, 70.0],
        "Condition": ["Norm_Norm", "Feedr_Norm"]
    }

    df = pd.DataFrame(data)
    return df.to_json(orient="split")