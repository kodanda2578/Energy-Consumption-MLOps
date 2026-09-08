"""
Data Loading Module for Energy Consumption Prediction Pipeline.

Provides reproducible dataset acquisition and loading functionality.
"""

import os
import io
import zipfile
import urllib.request
import pandas as pd

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file or return defaults if PyYAML is uninstalled."""
    default_config = {
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "raw_filename": "energydata_complete.csv",
            "processed_filename": "energydata_processed.csv",
            "url": "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip",
            "target_column": "Appliances",
            "datetime_column": "date",
        }
    }

    if HAS_YAML and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    return default_config


def download_raw_data(config_path: str = "configs/config.yaml") -> str:
    """
    Download the dataset from the UCI Machine Learning Repository
    and extract it into the raw data directory.

    Returns:
        str: Path to the downloaded raw CSV file.
    """
    config = load_config(config_path)
    raw_dir = config["data"]["raw_dir"]
    raw_filename = config["data"]["raw_filename"]
    dataset_url = config["data"]["url"]

    os.makedirs(raw_dir, exist_ok=True)
    target_csv_path = os.path.join(raw_dir, raw_filename)

    if os.path.exists(target_csv_path):
        print(f"[INFO] Raw dataset already exists at: {target_csv_path}")
        return target_csv_path

    print(f"[INFO] Downloading dataset from: {dataset_url}")
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(dataset_url, headers=headers)
    
    with urllib.request.urlopen(req) as response:
        zip_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        if raw_filename in z.namelist():
            z.extract(raw_filename, path=raw_dir)
            print(f"[SUCCESS] Extracted '{raw_filename}' into '{raw_dir}'")
        else:
            csv_files = [f for f in z.namelist() if f.endswith(".csv")]
            if csv_files:
                extracted_name = csv_files[0]
                z.extract(extracted_name, path=raw_dir)
                extracted_path = os.path.join(raw_dir, extracted_name)
                if extracted_name != raw_filename:
                    os.rename(extracted_path, target_csv_path)
                print(f"[SUCCESS] Extracted '{extracted_name}' and saved to '{target_csv_path}'")
            else:
                raise FileNotFoundError("No CSV file found inside downloaded ZIP archive.")

    return target_csv_path


def load_raw_data(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    """
    Load raw energy consumption data into a pandas DataFrame.

    Returns:
        pd.DataFrame: Raw dataset with parsed datetime.
    """
    config = load_config(config_path)
    raw_dir = config["data"]["raw_dir"]
    raw_filename = config["data"]["raw_filename"]
    target_csv_path = os.path.join(raw_dir, raw_filename)

    if not os.path.exists(target_csv_path):
        print(f"[INFO] Dataset not found locally. Initiating download...")
        target_csv_path = download_raw_data(config_path)

    df = pd.read_csv(target_csv_path)
    datetime_col = config["data"]["datetime_column"]

    if datetime_col in df.columns:
        df[datetime_col] = pd.to_datetime(df[datetime_col])

    print(f"[INFO] Successfully loaded raw dataset. Shape: {df.shape}")
    return df


if __name__ == "__main__":
    df_raw = load_raw_data()
    print("\nDataset Info:")
    print(df_raw.info())
    print("\nFirst 3 Rows:")
    print(df_raw.head(3))
