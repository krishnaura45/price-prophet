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

### FILES

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

### IMPORTANT LINKS

- 📂 Dataset Utilized (Kaggle): [Ames Housing Dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)
- 📄 Original Dataset (Kaggle): [House Prices- Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
- 📓 Notebook (Kaggle): [House Pricing EDA and Extensive Modeling](https://www.kaggle.com/code/krishd123/house-pricing-eda-and-extensive-modeling)

---

### INTRODUCTION
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
![image](https://github.com/KD-Blitz/PotatoLife/assets/118080140/2cf49bb6-f4da-4457-b784-1043395233ba)
-->
