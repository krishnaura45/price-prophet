import pandas as pd
from src.ingest_data import DataIngestorFactory
from zenml import step


@step
def data_ingestion_step(file_path: str) -> pd.DataFrame:
    """Ingest data from a ZIP file using the appropriate DataIngestor."""
    # Determining the file extension
    file_extension = ".zip"  # Since we're dealing with ZIP files, this is hardcoded

    # Getting the appropriate DataIngestor
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    # Ingesting the data and loading it into a DataFrame
    df = data_ingestor.ingest(file_path)
    return df
