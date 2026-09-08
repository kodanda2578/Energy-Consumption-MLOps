"""
Unit tests for data preprocessing and feature engineering module.
"""

import numpy as np
import pandas as pd
from src.preprocessing import preprocess_data, get_feature_target_split, chronological_split


def test_preprocessing_pipeline():
    df = preprocess_data()
    assert not df.empty
    
    # Check feature columns present
    expected_features = [
        "hour", "day_of_week", "month", "day", "is_weekend",
        "sin_hour", "cos_hour", "sin_day_of_week", "cos_day_of_week",
        "appliances_lag_1", "appliances_lag_3", "appliances_lag_6", "appliances_lag_12",
        "rolling_mean_3", "rolling_mean_6", "rolling_mean_12"
    ]
    for col in expected_features:
        assert col in df.columns, f"Missing feature column: {col}"

    # Verify no NaN values exist after preprocessing dropna
    assert df.isnull().sum().sum() == 0


def test_no_data_leakage_in_lags_and_rolling():
    """
    STRICT DATA LEAKAGE TEST:
    Verify that rolling_mean_3 at row i equals the average of past Appliances
    at row i-1, i-2, i-3, and DOES NOT include Appliances at current row i.
    """
    df = preprocess_data()
    
    # Check row index 10
    idx = 10
    actual_rolling_3 = df.loc[idx, "rolling_mean_3"]
    expected_past_mean = df.loc[idx-3:idx-1, "Appliances"].mean()
    
    # Assert rolling mean uses STRICTLY past values
    assert np.isclose(actual_rolling_3, expected_past_mean), (
        f"Data leakage detected! Expected rolling mean {expected_past_mean}, got {actual_rolling_3}"
    )


def test_chronological_split():
    df = preprocess_data()
    X, y, feature_names = get_feature_target_split(df)
    X_train, X_test, y_train, y_test = chronological_split(X, y, train_ratio=0.8)

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    assert len(X_train) == int(len(X) * 0.8)
