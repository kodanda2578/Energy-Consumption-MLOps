"""
Unit tests for model training, evaluation, and artifact serialization.
"""

import os
import joblib
import numpy as np
import pandas as pd
from src.preprocessing import preprocess_data, get_feature_target_split, chronological_split
from src.train import train_models


def test_model_training_and_serialization():
    artifact, results_df = train_models()

    assert os.path.exists("models/energy_predictor.joblib")
    assert not results_df.empty
    assert "RMSE" in results_df.columns
    assert "MAE" in results_df.columns
    assert "R2" in results_df.columns


def test_saved_model_prediction():
    artifact_path = "models/energy_predictor.joblib"
    assert os.path.exists(artifact_path), "Model artifact does not exist."

    artifact = joblib.load(artifact_path)
    assert "pipeline" in artifact
    assert "feature_names" in artifact

    pipeline = artifact["pipeline"]
    feature_names = artifact["feature_names"]

    # Preprocess a sample record for inference verification
    df = preprocess_data()
    X, _, _ = get_feature_target_split(df)
    sample_x = X.iloc[:5][feature_names]

    preds = pipeline.predict(sample_x)
    assert len(preds) == 5
    assert isinstance(preds, np.ndarray)
    assert not np.isnan(preds).any()
