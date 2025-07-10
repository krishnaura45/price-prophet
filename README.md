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
<img width="1636" height="609" alt="image" src="https://github.com/user-attachments/assets/498a0db6-45a9-4a75-bef9-061e5bcbe73e" />
<img width="1622" height="473" alt="image" src="https://github.com/user-attachments/assets/1128d23d-832d-487f-aba0-313071bdc34e" />
-->

### PROBLEM DEFINITION
- **Manual Workflows**: Traditional house price prediction lacks automation, requiring repetitive preprocessing, model training, and evaluation steps.
- **Pipeline Gaps**: Most ML solutions stop at model accuracy, missing crucial components like deployment, tracking, and maintainability.
- **Lack of Production Readiness**: Existing approaches don't support reproducible, scalable, or monitorable model deployment in real-world settings.
- **End-to-End MLOps**: There is a clear need for a robust, automated pipeline integrating data handling, modeling, versioning, and serving with real-time inference.

---

### OBJECTIVES
- **Ultimate Aim**: Build an end-to-end MLOps pipeline.
- Perform robust data processing and heavy feature engineering so as to get best model performance.
- Utilize and compare multiple regression strategies for price prediction.
- Integrate MLOps tools like ZenML and Mlflow.
- Build a front-end application for user interaction and visualization.
- Eensure production readiness by focusing on modularity, reproducibility, version control, and real-time prediction capability.

---
<!--
### METHODOLOGY
#### Pipeline Workflow

#### Key Stages

#### MLOps Stack

---

### PROPOSED FRAMEWORK
![image](https://github.com/KD-Blitz/PotatoLife/assets/118080140/c0f5cebc-a9d3-4870-81b5-8abd416b20c0)

---

### RESULTS
- Upon completion of training, our model achieved a remarkable test accuracy of 98.83%, demonstrating the model's efficacy in classifying potato leaves into three categories: early blight, late blight, and healthy conditions.
![image](https://github.com/KD-Blitz/PotatoLife/assets/118080140/d4a77d8c-e31f-41f6-b59f-0fce38e70c95)

- Alongside, we have the model’s predictions on a sample of test images, where:
  - “Actual" represents the true class.
  - “Predicted represents the predicted class.
  - “Confidence" is the confidence score associated with the prediction, which reflects upon the reliability of the model's decision.
![image](https://github.com/KD-Blitz/PotatoLife/assets/118080140/199ceb75-6bd8-4ab3-9f99-a75f5dcf4ddb)

- FastAPI Working
![image](https://github.com/KD-Blitz/PotatoLife/assets/118080140/7798b018-aef7-4b34-b0af-c73fae61cb29)
![image](https://github.com/KD-Blitz/PotatoLife/assets/118080140/2443d89d-37e4-423d-af33-11f3fdf755dd)


- Postman software interacting with the model via API request and giving prediction result as a json response for a sample leaf image.
![image](https://github.com/KD-Blitz/PotatoLife/assets/118080140/349da7dc-f94e-420e-a528-1d6fbfc74606)

- Web App Interface
![wai_1](https://github.com/user-attachments/assets/d27e7d67-fa97-4801-89b9-d625612d9930)
![wai_2](https://github.com/user-attachments/assets/ff2c23a3-9178-455f-8b6e-cb314c5a3cf2)

---

### INSTALLATION

To set up the project on your local machine, follow these steps:

1. **Clone the repository**:
```sh
    git clone https://github.com/krishnaura45/PotatoLife.git
    cd PotatoLife
```
2. **Install dependencies**:
```sh
    pip install -r requirements.txt
```
3. **Run the Streamlit app**:
```sh
    streamlit run app.py
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

- **Python**: Backend logic and data handling.
- **TensorFlow**: Deep learning model.
- **Streamlit**: Web app development.
- **FastAPI**: API interaction and server handling.
- **Postman**: API testing and interaction.
- **Pillow**, **NumPy**, **Matplotlib**: Image processing and visualization.

<!--
### CONCLUSIONS/OUTCOMES
- Robust Data Preparation: Data preprocessing, involving batch loading, resizing, rescaling, and data augmentation, enhanced dataset quality and diversity.
- Model achieved an impressive 98.83% test accuracy, showcasing deep learning's effectiveness in disease classification.
- User-Friendly Implementation: FastAPI and Postman software facilitated user-friendly disease classification, empowering farmers and experts for on-the-spot diagnosis.
- Effectively address timely and accurate detection of potato leaf diseases.
-->
---

### FUTURE PLANS
- Create a mobile app
- Emphasize on the need for FastAPI server security, including integration with Docker and measures like authentication and authorization to safeguard model and user data.

---

### REFERENCES
1) S. Biswas, B. Jagyasi, B. P. Singh and M. Lal, "Severity identification of Potato Late Blight disease from crop images captured under uncontrolled environment," 2014 IEEE Canada International Humanitarian Technology Conference - (IHTC), Montreal, QC, Canada, 2014, pp. 1-5, doi: 10.1109/IHTC.2014.7147519.
2) G. Athanikar and P. Badar, “Potato Leaf Diseases Detection and Classification System”, IJCSMC, Vol. 5, Issue. 2, February 2016, pg.76 – 88
3) Rabbia Mahum, Haris Munir, Zaib-Un-Nisa Mughal, Muhammad Awais, Falak Sher Khan, Muhammad Saqlain, Saipunidzam Mahamad & Iskander Tlili. A novel framework for potato leaf disease detection using an efficient deep learning model, Human and Ecological Risk Assessment: An International Journal; 2022.
4) Chakraborty KK, Mukherjee R, Chakroborty C, Bora K. Automated recognition of optical image based potato leaf blight diseases using deep learning. Physiological and Molecular Plant Pathology. 2022;117:101781

---

### Contributors 
- Krishna Dubey (Data Collection / Machine Learning / API Development / Web App Development)
- Pankaj Kumar Giri (API Code Help)
