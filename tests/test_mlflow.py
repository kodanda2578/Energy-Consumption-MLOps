"""
Unit & Integration tests for MLflow experiment tracking and model registry.
"""

import os
import mlflow
import pandas as pd
import numpy as np
from src.train import train_models
from src.preprocessing import preprocess_data, get_feature_target_split
from src.mlflow_utils import load_registered_model


def test_mlflow_training_and_tracking():
    """
    Verify that executing training creates MLflow runs, logs parameters, metrics,
    artifacts, and registers the best model under EnergyConsumptionModel.
    """
    artifact, results_df = train_models()

    assert artifact is not None
    assert not results_df.empty
    assert "mlflow_run_id" in artifact
    assert "registered_model_version" in artifact

    client = mlflow.tracking.MlflowClient()
    experiment = mlflow.get_experiment_by_name("Energy Consumption Prediction")
    assert experiment is not None

    # Check runs in experiment
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) >= 3, "Expected at least 3 model runs logged in MLflow."

    # Check best registered model
    registered_model_name = "EnergyConsumptionModel"
    model_version = client.get_model_version_by_alias(registered_model_name, "champion")
    assert model_version is not None
    assert model_version.name == registered_model_name


def test_registered_model_inference():
    """
    Verify loading the registered model from MLflow Model Registry via alias 'champion'
    and running inference on sample features.
    """
    model = load_registered_model(model_name="EnergyConsumptionModel", alias="champion")
    assert model is not None

    # Load sample features for inference
    df = preprocess_data()
    X, _, feature_names = get_feature_target_split(df)
    sample_x = X.iloc[:5][feature_names]

    preds = model.predict(sample_x)
    assert len(preds) == 5
    assert isinstance(preds, (np.ndarray, pd.Series))
    assert not np.isnan(preds).any()

