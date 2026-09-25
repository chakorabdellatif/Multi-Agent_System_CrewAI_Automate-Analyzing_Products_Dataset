import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from src.ai_analysis.tools.data_manager import DataManager


# -------------------- LoadDataTool --------------------
class LoadDataInput(BaseModel):
    path: str = Field(default="data/Products.csv", description="Path to raw data CSV file")


class LoadDataTool(BaseTool):
    name: str = "Load Data Tool"
    description: str = "Loads the raw fashion dataset from specified path"
    args_schema: type[BaseModel] = LoadDataInput

    def _run(self, path: str = "data/Products.csv") -> str:
        try:
            DataManager.load_csv(path)
            df = DataManager.get_df(processed=False)
            return f"Raw dataset loaded with {len(df)} rows from {path}"
        except Exception as e:
            return f"Data loading failed: {e!s}"


# -------------------- DataCleanerTool --------------------
class CleanDataInput(BaseModel):
    remove_na: bool = Field(default=True, description="Remove missing values")
    remove_duplicates: bool = Field(default=True, description="Remove duplicate rows")


class DataCleanerTool(BaseTool):
    name: str = "Data Cleaner Tool"
    description: str = "Cleans data by handling missing values and duplicates"
    args_schema: type[BaseModel] = CleanDataInput

    def _run(self, remove_na: bool = True, remove_duplicates: bool = True) -> str:
        try:
            df = DataManager.get_df(processed=False)
            initial_count = len(df)

            if remove_na:
                df = df.dropna()
            if remove_duplicates:
                df = df.drop_duplicates()

            DataManager.set_df(df)
            return (
                f"Cleaned data: {initial_count} → {len(df)} rows "
                f"(NA: {'removed' if remove_na else 'kept'}, "
                f"Dups: {'removed' if remove_duplicates else 'kept'})"
            )
        except Exception as e:
            return f"Data cleaning failed: {e!s}"


# -------------------- OutlierRemoverTool --------------------
class RemoveOutliersInput(BaseModel):
    columns: list[str] = Field(default=["Price", "Rating"], description="Numeric columns for outlier removal")
    iqr_factor: float = Field(default=1.5, description="IQR multiplier for outlier detection")


class OutlierRemoverTool(BaseTool):
    name: str = "Outlier Remover Tool"
    description: str = "Removes outliers using IQR method from specified columns"
    args_schema: type[BaseModel] = RemoveOutliersInput

    def _run(self, columns: list[str] | None = None, iqr_factor: float = 1.5) -> str:
        if columns is None:
            columns = ["Price", "Rating"]
        try:
            df = DataManager.get_df(processed=True)
            initial_count = len(df)

            for col in columns:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    mask = (df[col] >= q1 - iqr_factor * iqr) & (df[col] <= q3 + iqr_factor * iqr)
                    df = df[mask]

            DataManager.set_df(df)
            removed = initial_count - len(df)
            return f"Outlier removal: {removed} rows removed from {columns}"
        except Exception as e:
            return f"Outlier removal failed: {e!s}"
