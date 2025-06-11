# Based on **Factory Design Pattern**
import os
import zipfile
from abc import ABC, abstractmethod
import pandas as pd

# Defining an abstract class for Data Ingestor
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Abstract method to ingest data from a given file."""

        pass


# Implementing a concrete class for ZIP Ingestion
class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Extracts a .zip file and returns the content as a pandas DataFrame."""

        # Ensure that the file is a .zip file
        if not file_path.endswith(".zip"):
            raise ValueError("The provided file is not a .zip file.")

        # Extract the zip file
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall("extracted_data")

        # Find the extracted CSV file (assuming there is one CSV file inside the zip)
        extracted_files = os.listdir("extracted_data")
        csv_files = [f for f in extracted_files if f.endswith(".csv")]

        # Handling 0 or multiple files
        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV file found in the extracted data.")
        if len(csv_files) > 1:
            raise ValueError("Multiple CSV files found. Please specify which one to use.")

        # Read the CSV into a DataFrame
        csv_file_path = os.path.join("extracted_data", csv_files[0])
        df = pd.read_csv(csv_file_path)

        # Return the DataFrame
        return df

# Implementing other concrete classes for JSON, EXCEL, PARQUET and XML Ingestion
class JsonDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Reads a .json file and returns the content as a pandas DataFrame."""
        
        if not file_path.endswith(".json"):
            raise ValueError("The provided file is not a .json file.")
        return pd.read_json(file_path)
    
class ExcelDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Reads a .xlsx file and returns the content as a pandas DataFrame."""
        
        if not file_path.endswith(".xlsx"):
            raise ValueError("The provided file is not a .xlsx file.")
        return pd.read_excel(file_path)

class ParquetDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Reads a .parquet file and returns the content as a pandas DataFrame."""
        
        if not file_path.endswith(".parquet"):
            raise ValueError("The provided file is not a .parquet file.")
        return pd.read_parquet(file_path)

class XmlDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Reads a .xml file and returns the content as a pandas DataFrame."""

        if not file_path.endswith(".xml"):
            raise ValueError("The provided file is not a .xml file.")
        return pd.read_xml(file_path)

# Implementing a Factory to create DataIngestors
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        """Returns the appropriate DataIngestor based on file extension."""

        if file_extension == ".zip":
            return ZipDataIngestor()
        elif file_extension == ".json":
            return JsonDataIngestor()
        elif file_extension == ".xlsx":
            return ExcelDataIngestor()
        elif file_extension == ".parquet":
            return ParquetDataIngestor()
        elif file_extension == ".xml":
            return XmlDataIngestor()
        else:
            raise ValueError(f"No ingestor available for file extension: {file_extension}")


# Using the factory to execute ingestion
if __name__ == "__main__":
    file_path = "/Users/Krishna/Downloads/Priceprophet/price-prophet/data/archive.zip"

    # Determine the file extension
    file_extension = os.path.splitext(file_path)[1]

    # Get the appropriate DataIngestor
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    # Ingest the data and load it into a DataFrame
    df = data_ingestor.ingest(file_path)

    # Now df contains the DataFrame from the extracted CSV
    print(df.head())

# Type check-ins
# Readability prioritized
# Error Handling