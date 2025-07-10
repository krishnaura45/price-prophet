<h1 align="center">Price Prophet: An end-to-end MLOps Pipeline for House Price Prediction</h1>

<b>Price Prophet</b> is a production-grade machine learning pipeline built to predict house sale prices using a robust, scalable and reproducible MLOps framework.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
![ZenML](https://img.shields.io/badge/ZenML-FF4088?style=for-the-badge&logo=zenml&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-FFBB00?style=for-the-badge&logo=mlflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

<!--
![MLOps](https://img.shields.io/badge/MLOps-End--to--End--Pipeline-brightgreen?style=for-the-badge)
![Optimized for RMSE](https://img.shields.io/badge/Optimized--For-RMSE-yellow?style=for-the-badge) 
-->

---

### FILES & STRUCTURE 📂

- `data/`: Raw zipped data from Kaggle
- `extracted_data/`: Ingested dataset
- `analysis/`: Exploratory notebooks and analyzers
- `src/`: Core modules for feature engineering, model building, evaluation
- `steps/`: ZenML-defined step-wise modular pipeline stages
- `pipelines/`: Training and deployment pipeline definitions
- `run_pipeline.py`: Executes training pipeline
- `run_deployment.py`: Executes deployment/inference pipeline
- `streamlit_app.py`: Streamlit interface for user-side predictions
- `expected_columns.json`: Logged expected columns for model inference
- `sample_predict.py`: Local REST inference - single sample
- `sample_batch_predict.py`: Local REST inference - batch prediction
- `exported_model/`: Model artifacts manually saved via MLflow
- `requirements.txt`: Python dependencies

---

### IMPORTANT LINKS 🔗

- 📂 Dataset Utilized (Kaggle): [Ames Housing Dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)
- 📄 Original Dataset (Kaggle): [House Prices- Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
- 📓 Notebook (Kaggle): [House Pricing EDA and Extensive Modeling](https://www.kaggle.com/code/krishd123/house-pricing-eda-and-extensive-modeling)

---

### INTRODUCTION

<img align="right" width="320" height="300" alt="house-image" src="https://github.com/user-attachments/assets/5708a76c-8e57-48c7-8d7b-8284ddf3b209" />

- Accurate house price prediction is vital for real estate valuation, investment, and decision-making.
- Traditional ML workflows often suffer from:
  - Poor reproducibility and pipeline modularity
  - Lack of production-readiness and deployment integration
  - Minimal tracking or model lifecycle management
- Price Prophet addresses this by building a clean, reproducible, and production-ready ML pipeline from ingestion to deployment.
- Built with Python, ZenML, MLflow, and Streamlit, it ensures seamless orchestration, experiment tracking, deployment, and user-friendly inference.

---
<!--
### RELATED WORKS
<img width="1636" height="609" alt="image" src="https://github.com/user-attachments/assets/498a0db6-45a9-4a75-bef9-061e5bcbe73e" />
<img width="1622" height="473" alt="image" src="https://github.com/user-attachments/assets/1128d23d-832d-487f-aba0-313071bdc34e" />
-->

### PROBLEM DEFINITION
- **Manual Workflows**: Traditional house price prediction lacks automation, requiring repetitive preprocessing, model training, and evaluation steps.
- **Pipeline Gaps**: Most ML solutions stop at model accuracy, missing crucial components like deployment, tracking, and maintainability.
- **Lack of Production Readiness**: Existing approaches don't support reproducible, scalable, or monitorable model deployment in real-world settings.
- **End-to-End MLOps**: There is a clear need for a robust, automated pipeline integrating data handling, modeling, versioning, and serving with real-time inference.

---

### OBJECTIVES 🧰
- **Ultimate Aim**: Build an end-to-end MLOps pipeline.
- Perform robust data processing and heavy feature engineering so as to get best model performance.
- Utilize and compare multiple regression strategies for price prediction.
- Integrate MLOps tools like ZenML and Mlflow.
- Build a front-end application for user interaction and visualization.
- Eensure production readiness by focusing on modularity, reproducibility, version control, and real-time prediction capability.

---

### METHODOLOGY 🔧
#### Pipeline Workflow
<img width="1538" height="692" alt="image" src="https://github.com/user-attachments/assets/8e4fbe08-842c-4081-937a-c92cad53733f" />

#### Core ML Stages
| **Stage**                     | **Description**                                                                                                                                                                                                                                                                                                     |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Data Ingestion**         | Raw housing dataset is loaded.                                                                                                                                                                                                        |
| **2. Initial Preprocessing** | Raw data is pre-processed: ID features dropped, duplicates removed, and indexing reset. Categorical and numerical missing values handled using strategy-based imputers (`SimpleImputer`), with special treatment for string `'None'` values.                                                                                                                                                                |
| **3. Heavy Feature Engineering**    | - Applied `log1p` transformation to skewed features<br>- Created engineered features: `Bath_total`, `BsmtFinSF`, `Porch`, `FinSF`<br>- Applied ordinal encoding on quality-type categorical columns<br>- Handled rare categories and normalized dirty labels<br>- Imputed `Lot Frontage` via trained XGBoost model. |
| **4. Outlier Handling**       | Explicit step skipped — engineered transformations addressed noise and anomalies effectively.                                                                                                                                                                                                                       |
| **5. Data Splitting**         | Dataset split into `train` and `test` using `train_test_split` (80:20) based on target column `SalePrice`.                                                                                                                                                                                                          |
| **6. Model Training**         | Trained initial `CatBoostRegressor` using categorical feature indices for native support, tuned for RMSE loss.                                                                                                                                                                                                      |
| **7. Hyperparameter Tuning**  | Used Optuna optimization framework over 50 trials for CatBoost hyperparameter tuning.                                                                                                                                                                                                                               |
| **8. Final Model**            | Trained CatBoost model using best params: `learning_rate`, `depth`, `l2_leaf_reg`, `bagging_temperature`, etc. with 1000 iterations.                                                                                                                                                                                |
| **9. Evaluation**             | Final model evaluated on `test set` using `MSE`, `RMSE`, and `R²` metrics.                                                                                                                                                                                                                                          |
| **10. Inference Integration** | Model predictions served using MLflow REST API and tested with scripts (`sample_predict.py`, `sample_batch_predict.py`) and Streamlit app.                                                                                                                                                                          |


#### MLOps Stack
<img width="1590" height="719" alt="image" src="https://github.com/user-attachments/assets/4f1ab329-751e-49c9-a8a3-6778be4f4967" />

#### Model Deployment
- Deployment via MLflow Model Deployer Service (Not suitable for Windows OS)<img width="307" height="111" alt="image" src="https://github.com/user-attachments/assets/3d7e8fc6-e88a-4203-a9ed-116208bddc96" />

- Manual MLflow Model Serving via REST API (Works for MAC/Windows OS)<img width="408" height="121" alt="image" src="https://github.com/user-attachments/assets/514fefad-9a26-477d-8924-c0612f2fcb4c" />

#### Inference
- **Batch Inference (Local REST API)**:
  - Once the model is served manually using MLflow, predictions can be made by sending input data (as JSON) via HTTP POST to the /invocations endpoint.
  - A sample_batch_predict.py script is used to load a .csv file, send data to the model server, and save predictions in predictions.csv.

- **Real-Time Inference (Streamlit Application)**:
  - A user-friendly UI built with Streamlit allows manual input or CSV uploads.
  - Sends the data to the same REST endpoint and displays predicted house prices instantly.
  - Supports downloading predictions and visualization inside the web app.

---
<!--
### PROPOSED FRAMEWORK
![image](https://github.com/KD-Blitz/PotatoLife/assets/118080140/c0f5cebc-a9d3-4870-81b5-8abd416b20c0)
-->

### RESULTS 📊
- High Level EDA <img width="1919" height="1008" alt="image" src="https://github.com/user-attachments/assets/7f32d9ee-a319-44d1-9fe5-1c679d28add5" />

- Comparative Model Evaluation Metrics <img width="1391" height="546" alt="image" src="https://github.com/user-attachments/assets/0541f369-bd35-4248-9ecc-61b0a1ae1bc3" />


- MLflow UI ~ Continuous Training and Deployment Run Details (for CatBoost) <img width="1403" height="715" alt="image" src="https://github.com/user-attachments/assets/e21629ba-54df-43f2-bded-0fd8ce8aaf6e" />

- Streamlit Application (Interface) <img width="1506" height="748" alt="image" src="https://github.com/user-attachments/assets/8f028ea5-3ce4-4d51-9359-1d5b2168d27d" />

- Making Predictions on App using second mode <img width="1575" height="691" alt="image" src="https://github.com/user-attachments/assets/beb5899a-1c2a-47a1-af98-3cc22c07ea0d" />

---

### INSTALLATION 🤖

To set up the project on your local machine, follow these steps:

1. **Clone the repository**:
```sh
    https://github.com/krishnaura45/price-prophet.git
    cd price-prophet
```
2. **Install dependencies**:
```sh
    pip install -r requirements.txt
```
3. **Run training pipeline**:
```sh
    python run_pipeline.py
```
4. **Serve model manually (use MLflow UI to fetch run ID)**:
```sh
    mlflow models serve -m "runs:/<your_run_id>/model" -p 1234 --no-conda
```
5. **Run deployment pipeline**:
```sh
    python run_deployment.py
```
6. **Run the Streamlit app**:
```sh
    streamlit run streamlit_app.py
```

---

### CONTRIBUTING
- Fork the repository.
- Create a new branch.
- Commit changes with clear messages.
- Submit a pull request.
- Ensure new features are tested and documented.

---

### TECH STACK
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white) 
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-Regressor-orange?style=for-the-badge)

### FEATURES 🚀
- 🔄 Modular ZenML Steps (each in steps/)
- 🧐 Advanced EDA and feature insights (analysis/)
- 📊 Tuned CatBoost for optimal accuracy
- 🪤 Model evaluation with proper metrics
- 🚪 Manual model deployment (you control what gets served)
- 🔗 Streamlit App for UI-based input, visualization and download

---

### FUTURE SCOPE 🔮
- **Cloud-Native Deployment**: Containerize the pipeline using Docker and orchestrate via Kubernetes to enable scalable, consistent, and production-ready deployments across cloud platforms.
- **Drift Detection & AutoML**: Implement data drift monitoring (e.g., with Evidently/WhyLabs) and integrate AutoML frameworks for continual model retraining and optimization.
- **Model Explainability**: Enhance interpretability using SHAP or LIME and display visual explanations in Streamlit for better decision trust and transparency.

---

### REFERENCES
1) ZenML Docs - https://docs.zenml.io/
2) MLflow Docs - https://mlflow.org/docs/latest/index.html
3) CatBoost Documentation - https://catboost.ai/en/docs/

---

### Contributors 🧑‍💼
- Krishna Dubey (Pipeline design, ML modeling, deployment, UI dev)
