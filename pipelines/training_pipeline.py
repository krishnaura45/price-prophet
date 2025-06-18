from steps.data_ingestion_step import data_ingestion_step
from steps.data_splitting_step import data_splitting_step
from steps.feature_engineering_step import feature_engineering_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.model_building_step import model_building_step
from steps.model_evaluation_step import model_evaluation_step
from steps.outlier_handling_step import outlier_handling_step
from zenml import Model, pipeline, step


@pipeline(
    model=Model(
        # The name uniquely identifies this model
        name="prices_predictor"
    ),
)
def ml_pipeline():
    """Define an end-to-end machine learning pipeline."""

    # Data Ingestion Step
    raw_data = data_ingestion_step(
        file_path="data/archive.zip"
    )

    # Handling Missing Values Step
    filled_data = handle_missing_values_step(raw_data)

    # Feature Engineering Step
    engineered_data = feature_engineering_step(
        filled_data, strategy="log", features=["Gr Liv Area", "SalePrice"]
    )

    # Outlier Detection Step
    clean_data = outlier_handling_step(engineered_data, column_name="SalePrice")

    # Data Splitting Step
    X_train, X_test, y_train, y_test = data_splitting_step(clean_data, target_column="SalePrice")

    # Model Building Step
    model = model_building_step(X_train=X_train, y_train=y_train)

    # Model Evaluation Step
    evaluation_metrics, mse = model_evaluation_step(
        trained_model=model, X_test=X_test, y_test=y_test
    )

    return model


if __name__ == "__main__":
    # Running the pipeline
    run = ml_pipeline()
