"""
Model Training and Selection Module for Energy Consumption Pipeline.

Trains baseline models (Linear Regression, Random Forest, XGBoost),
evaluates metrics on test set, selects the best performing model,
and serializes the full pipeline artifact using joblib.
"""

import sys
import os

# Ensure root directory is in sys.path for direct invocation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.data_loader import load_config
from src.preprocessing import preprocess_data, get_feature_target_split, chronological_split
from src.evaluate import evaluate_predictions, generate_evaluation_plots, generate_feature_importance_plot
from src.mlflow_utils import setup_mlflow_experiment, log_model_run, register_best_model


def train_models(config_path: str = "configs/config.yaml"):
    """
    Execute full training, evaluation, MLflow tracking, and model registration workflow.
    """
    config = load_config(config_path)
    train_ratio = config.get("split", {}).get("train_ratio", 0.8)
    random_state = config.get("model", {}).get("random_state", 42)
    save_dir = config.get("model", {}).get("save_dir", "models")
    model_filename = config.get("model", {}).get("model_filename", "energy_predictor.joblib")

    # 1. Preprocess Data & Chronological Split
    df = preprocess_data(config_path)
    X, y, feature_names = get_feature_target_split(df, config_path)
    X_train, X_test, y_train, y_test = chronological_split(X, y, train_ratio=train_ratio)

    dataset_info = {
        "dataset_name": "Appliances Energy Prediction",
        "target_column": config.get("data", {}).get("target_column", "Appliances"),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "feature_count": X.shape[1]
    }

    # 2. Setup MLflow Experiment
    setup_mlflow_experiment("Energy Consumption Prediction")

    # 3. Define Model Candidates & Parameters
    rf_params = {
        "n_estimators": config["model"].get("rf_n_estimators", 100),
        "max_depth": config["model"].get("rf_max_depth", 15),
        "min_samples_split": 2,
        "random_state": random_state,
        "n_jobs": -1
    }
    
    xgb_params = {
        "n_estimators": config["model"].get("xgb_n_estimators", 100),
        "learning_rate": config["model"].get("xgb_learning_rate", 0.05),
        "max_depth": config["model"].get("xgb_max_depth", 6),
        "subsample": 1.0,
        "random_state": random_state,
        "n_jobs": -1
    }

    candidates = {
        "Linear Regression": (
            Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
            {"model_type": "Linear Regression"}
        ),
        "Random Forest": (
            Pipeline([("scaler", StandardScaler()), ("model", RandomForestRegressor(
                n_estimators=rf_params["n_estimators"],
                max_depth=rf_params["max_depth"],
                min_samples_split=rf_params["min_samples_split"],
                random_state=rf_params["random_state"],
                n_jobs=rf_params["n_jobs"]
            ))]),
            {
                "model_type": "Random Forest",
                "n_estimators": rf_params["n_estimators"],
                "max_depth": rf_params["max_depth"],
                "min_samples_split": rf_params["min_samples_split"],
                "random_state": rf_params["random_state"]
            }
        ),
        "XGBoost": (
            Pipeline([("scaler", StandardScaler()), ("model", XGBRegressor(
                n_estimators=xgb_params["n_estimators"],
                learning_rate=xgb_params["learning_rate"],
                max_depth=xgb_params["max_depth"],
                subsample=xgb_params["subsample"],
                random_state=xgb_params["random_state"],
                n_jobs=xgb_params["n_jobs"]
            ))]),
            {
                "model_type": "XGBoost",
                "n_estimators": xgb_params["n_estimators"],
                "learning_rate": xgb_params["learning_rate"],
                "max_depth": xgb_params["max_depth"],
                "subsample": xgb_params["subsample"],
                "random_state": xgb_params["random_state"]
            }
        )
    }

    # 4. Model Training & MLflow Experiment Tracking Loop
    results = {}
    trained_pipelines = {}
    run_ids = {}

    print("\n" + "=" * 60)
    print("      MODEL TRAINING AND MLFLOW TRACKING RESULTS")
    print("=" * 60)

    for name, (pipeline, params) in candidates.items():
        print(f"[INFO] Training & Tracking {name}...")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred)
        results[name] = metrics
        trained_pipelines[name] = pipeline

        # Generate diagnostic plots & feature importance plot
        figures_dir = config.get("reports", {}).get("figures_dir", "reports/figures")
        generate_evaluation_plots(y_test, y_pred, name, config_path)
        feat_img = generate_feature_importance_plot(pipeline.named_steps["model"], feature_names, name, config_path)

        # Log MLflow Run
        run_id = log_model_run(
            model_name=name,
            pipeline=pipeline,
            params=params,
            metrics=metrics,
            dataset_info=dataset_info,
            figures_dir=figures_dir,
            feature_importance_plot=feat_img
        )
        run_ids[name] = run_id
        print(f"       -> MLflow Run ID: {run_id} | RMSE: {metrics['RMSE']}")

    # Display Results Comparison Table
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values(by="RMSE", ascending=True)
    print("\nComparison Table:")
    print(results_df.to_string())

    # 5. Select Best Model (Lowest RMSE)
    best_model_name = results_df.index[0]
    best_pipeline = trained_pipelines[best_model_name]
    best_metrics = results[best_model_name]
    best_run_id = run_ids[best_model_name]

    print(f"\n[BEST MODEL SELECTED]: '{best_model_name}' (Run ID: {best_run_id}) achieved lowest RMSE: {best_metrics['RMSE']}")

    # 6. MLflow Model Registry
    reg_info = register_best_model(
        run_id=best_run_id,
        model_name="EnergyConsumptionModel",
        alias="champion"
    )

    print("\n" + "=" * 60)
    print("      REGISTERED MODEL DETAILS")
    print("=" * 60)
    print(f"  Model Name    : {reg_info['model_name']}")
    print(f"  Model Version : {reg_info['version']}")
    print(f"  Alias         : {reg_info['alias']}")
    print(f"  Run ID        : {reg_info['run_id']}")
    print(f"  RMSE          : {best_metrics['RMSE']}")
    print(f"  MAE           : {best_metrics['MAE']}")
    print(f"  R2            : {best_metrics['R2']}")
    print("=" * 60 + "\n")

    # 7. Serialize Local Pipeline Artifact (Phase 3 compatibility)
    os.makedirs(save_dir, exist_ok=True)
    artifact_path = os.path.join(save_dir, model_filename)

    artifact = {
        "pipeline": best_pipeline,
        "feature_names": feature_names,
        "model_name": best_model_name,
        "metrics": best_metrics,
        "config": config,
        "mlflow_run_id": best_run_id,
        "registered_model_version": reg_info['version']
    }

    joblib.dump(artifact, artifact_path)
    print(f"[SUCCESS] Serialized best pipeline artifact saved to: {artifact_path}\n")

    return artifact, results_df


if __name__ == "__main__":
    train_models()
