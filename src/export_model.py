"""
Model Export Script for Standalone Deployment.

Exports a verified registered MLflow model version (or alias) into a self-contained,
deployment-ready directory that can be loaded by FastAPI in production environments
without requiring local mlflow.db or mlruns directories.
"""

import os
import sys
import argparse
import pandas as pd
import mlflow
import mlflow.artifacts

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.mlflow_utils import setup_mlflow_experiment


def export_registered_model(
    model_name: str = "EnergyConsumptionModel",
    version: str = "30",
    output_dir: str = "deployment_model"
) -> str:
    """
    Download and export registered MLflow model version artifacts to a standalone directory.

    Args:
        model_name (str): Name of the registered model.
        version (str): Model version number or alias.
        output_dir (str): Target directory to save standalone model artifacts.

    Returns:
        str: Absolute path to the exported model directory.
    """
    output_dir_path = os.path.abspath(os.path.join(PROJECT_ROOT, output_dir))
    os.makedirs(output_dir_path, exist_ok=True)

    # Initialize MLflow tracking to read from local repository registry
    setup_mlflow_experiment("Energy Consumption Prediction")

    # Formulate model URI (e.g. models:/EnergyConsumptionModel/30)
    model_uri = f"models:/{model_name}/{version}"
    print(f"[INFO] Exporting registered model from URI: {model_uri} -> {output_dir_path}...")

    # Download artifacts directly using MLflow artifacts API
    exported_path = mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri,
        dst_path=output_dir_path
    )
    print(f"[SUCCESS] Model artifacts successfully exported to: {exported_path}")

    return exported_path


def verify_exported_model(model_dir: str) -> float:
    """
    Verify the exported standalone model by loading it and executing a sample prediction.

    Args:
        model_dir (str): Path to standalone model directory.

    Returns:
        float: Sample prediction value.
    """
    print(f"[INFO] Verifying exported model loading from: {model_dir}...")
    model = mlflow.pyfunc.load_model(model_dir)

    # Standard sample payload containing 41 features
    sample_payload = {
        "lights": 0.0,
        "T1": 19.89, "RH_1": 47.59,
        "T2": 19.2, "RH_2": 44.79,
        "T3": 19.79, "RH_3": 44.73,
        "T4": 19.0, "RH_4": 45.56,
        "T5": 17.1667, "RH_5": 55.2,
        "T6": 7.0267, "RH_6": 84.2567,
        "T7": 17.2, "RH_7": 41.6267,
        "T8": 18.2, "RH_8": 48.9,
        "T9": 17.0333, "RH_9": 45.53,
        "T_out": 6.6, "Press_mm_hg": 733.5, "RH_out": 92.0,
        "Windspeed": 7.0, "Visibility": 63.0, "Tdewpoint": 5.3,
        "hour": 18, "day_of_week": 0, "month": 1, "day": 11, "is_weekend": 0,
        "sin_hour": -1.0, "cos_hour": 0.0,
        "sin_day_of_week": 0.0, "cos_day_of_week": 1.0,
        "appliances_lag_1": 60.0, "appliances_lag_3": 50.0,
        "appliances_lag_6": 50.0, "appliances_lag_12": 60.0,
        "rolling_mean_3": 53.3333, "rolling_mean_6": 55.0, "rolling_mean_12": 58.3333
    }

    input_df = pd.DataFrame([sample_payload])
    predictions = model.predict(input_df)
    pred_val = float(predictions[0])
    print(f"[SUCCESS] Standalone model verification prediction: {pred_val:.4f} Wh")
    return pred_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MLflow registered model for standalone deployment")
    parser.add_argument("--model-name", default="EnergyConsumptionModel", help="Registered model name")
    parser.add_argument("--version", default="30", help="Model version number or alias (default: 30)")
    parser.add_argument("--output-dir", default="deployment_model", help="Directory for exported model")

    args = parser.parse_args()
    exported_dir = export_registered_model(
        model_name=args.model_name,
        version=args.version,
        output_dir=args.output_dir
    )
    verify_exported_model(exported_dir)
