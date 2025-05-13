import pandas as pd
import os
from pathlib import Path

class DataManager:
    _raw_df = None
    _processed_df = None
    _processed_path = Path("data/cleaned_products.csv")
    
    @classmethod
    def load_csv(cls, path: str) -> None:
        """Load raw data from CSV and initialize processed data"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        try:
            cls._raw_df = pd.read_csv(path)
            cls._processed_df = cls._raw_df.copy()
            print(f"✅ Successfully loaded raw data from {path}")
        except Exception as e:
            raise ValueError(f"Error loading CSV: {str(e)}")

    @classmethod
    def get_df(cls, processed: bool = True) -> pd.DataFrame:
        """Get the current dataframe version"""
        df = cls._processed_df if processed else cls._raw_df
        if df is None:
            raise ValueError("No data loaded. Use load_csv() first.")
        return df.copy()

    @classmethod
    def set_df(cls, df: pd.DataFrame) -> None:
        """Update and persist processed dataframe"""
        try:
            cls._processed_df = df.copy()
            cls._save_processed_data()
            print(f"💾 Saved processed data to {cls._processed_path}")
        except Exception as e:
            raise RuntimeError(f"Error saving processed data: {str(e)}")

    @classmethod
    def _save_processed_data(cls) -> None:
        """Internal method to persist processed data"""
        cls._processed_path.parent.mkdir(parents=True, exist_ok=True)
        cls._processed_df.to_csv(cls._processed_path, index=False)

    @classmethod
    def load_processed(cls) -> None:
        """Load processed data from disk"""
        try:
            if not cls._processed_path.exists():
                raise FileNotFoundError(f"Processed data not found at {cls._processed_path}")
            
            cls._processed_df = pd.read_csv(cls._processed_path)
            print(f"🔁 Loaded processed data from {cls._processed_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading processed data: {str(e)}")

    @classmethod
    def reset(cls) -> None:
        cls._raw_df = None
        cls._processed_df = None
        if cls._processed_path.exists():
            cls._processed_path.unlink()