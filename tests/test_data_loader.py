"""
Unit tests for data loading module.
"""

import os
import pandas as pd
from src.data_loader import load_raw_data, load_config


def test_config_loading():
    config = load_config()
    assert "data" in config
    assert config["data"]["target_column"] == "Appliances"
    assert config["data"]["datetime_column"] == "date"


def test_raw_data_loading():
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Appliances" in df.columns
    assert "date" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
