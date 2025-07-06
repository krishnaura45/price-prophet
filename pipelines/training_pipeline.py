from steps.data_ingestion_step import data_ingestion_step
from steps.data_splitting_step import data_splitting_step
from steps.feature_engineering_step import feature_engineering_step
from steps.initial_preprocessing_step import initial_preprocessing_step
from steps.model_building_step import model_building_step
from steps.model_evaluation_step import model_evaluation_step
from zenml import Model, pipeline


@pipeline(
    model=Model(
        name="prices_predictor_optimum"
    ),
)
def ml_pipeline():
    """Define an end-to-end machine learning pipeline."""

    # Data Ingestion Step
    raw_data = data_ingestion_step(
        file_path="C:/Users/Krishna/Downloads/major-project/price-prophet-final/data/archive.zip"
    )

    # Initial Preprocessing Step (dropping, de-duping, partial imputation)
    filled_data = initial_preprocessing_step(raw_data)

    # Feature Engineering Step (heavy engineered features + log SalePrice)
    engineered_data = feature_engineering_step(
        filled_data, features=["SalePrice"]
    )

    # Data Splitting Step
    X_train, X_test, y_train, y_test = data_splitting_step(
        engineered_data, target_column="SalePrice"
    )

    # Model Building Step (CatBoost Tuned Default)
    model = model_building_step(X_train=X_train, y_train=y_train)

    # Model Evaluation Step
    evaluation_metrics = model_evaluation_step(
        trained_model=model, X_test=X_test, y_test=y_test
    )

    return model


if __name__ == "__main__":
    run = ml_pipeline()