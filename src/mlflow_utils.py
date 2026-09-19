"""
MLflow Integration Module for Energy Consumption Pipeline.

Provides functions for experiment setup, logging model runs,
and managing the MLflow Model Registry and aliases.
"""

import sys
import os
import sqlite3
import mlflow
from mlflow.tracking import MlflowClient

# Ensure root directory is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# Project-root-anchored MLflow SQLite DB file path
MLFLOW_DB_PATH = os.path.join(PROJECT_ROOT, "mlflow.db")


def get_tracking_uri() -> str:
    """
    Get the deterministic MLflow tracking URI anchored to the project root.
    Formats as: sqlite:///... with POSIX slashes on both Windows and Linux.
    """
    normalized_path = os.path.abspath(MLFLOW_DB_PATH).replace("\\", "/")
    return f"sqlite:///{normalized_path}"


def get_db_path_from_uri(tracking_uri: str) -> str:
    """
    Extract the local file path from a sqlite:/// URI.
    """
    if tracking_uri.startswith("sqlite:///"):
        path = tracking_uri[len("sqlite:///")]
        return os.path.abspath(path)
    return MLFLOW_DB_PATH


def sanitize_mlflow_db_paths(db_path: str = None):
    """
    Ensure all artifact URIs and locations in MLflow SQLite database use relative paths,
    stripping environment-specific absolute path prefixes. This guarantees portability
    across OS environments (e.g. Windows native vs Linux Docker container).
    """
    if db_path is None:
        db_path = MLFLOW_DB_PATH

    if not os.path.exists(db_path):
        return

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # Update experiments table (e.g. file:C:/.../mlruns/1 or /home/.../mlruns/1 -> mlruns/1)
        cur.execute("SELECT experiment_id, artifact_location FROM experiments")
        for exp_id, loc in cur.fetchall():
            if loc and "mlruns" in loc and not loc.startswith("mlruns"):
                idx = loc.find("mlruns")
                if idx != -1:
                    new_loc = loc[idx:].replace("\\", "/")
                    cur.execute("UPDATE experiments SET artifact_location = ? WHERE experiment_id = ?", (new_loc, exp_id))

        # Update runs table
        cur.execute("SELECT run_uuid, artifact_uri FROM runs")
        for run_id, uri in cur.fetchall():
            if uri and "mlruns" in uri and not uri.startswith("mlruns"):
                idx = uri.find("mlruns")
                if idx != -1:
                    new_uri = uri[idx:].replace("\\", "/")
                    cur.execute("UPDATE runs SET artifact_uri = ? WHERE run_uuid = ?", (new_uri, run_id))

        # Update model_versions table
        cur.execute("SELECT name, version, storage_location FROM model_versions")
        for name, version, loc in cur.fetchall():
            if loc and "mlruns" in loc and not loc.startswith("mlruns"):
                idx = loc.find("mlruns")
                if idx != -1:
                    new_loc = loc[idx:].replace("\\", "/")
                    cur.execute("UPDATE model_versions SET storage_location = ? WHERE name = ? AND version = ?", (new_loc, name, version))

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[WARNING] MLflow DB path sanitization warning: {e}")


def setup_mlflow_experiment(experiment_name: str = "Energy Consumption Prediction", tracking_uri: str = None):
    """
    Configure MLflow tracking URI, registry URI, environment variables,
    and active experiment using the project-root anchored database.

    Args:
        experiment_name (str): Name of the MLflow experiment.
        tracking_uri (str, optional): Explicit tracking URI. Defaults to project-root anchored SQLite DB.

    Returns:
        str: Active experiment ID.
    """
    if tracking_uri is None:
        tracking_uri = get_tracking_uri()

    # 1. Set environment variables so all sub-modules and MLflow clients inherit the URI
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    # 2. Sanitize database paths if using SQLite
    if "sqlite" in tracking_uri:
        db_file = get_db_path_from_uri(tracking_uri)
        sanitize_mlflow_db_paths(db_file)

    # 3. Set MLflow tracking and registry URIs explicitly
    mlflow.set_tracking_uri(tracking_uri)
    try:
        mlflow.set_registry_uri(tracking_uri)
    except Exception:
        pass

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)

    if "sqlite" in tracking_uri:
        db_file = get_db_path_from_uri(tracking_uri)
        sanitize_mlflow_db_paths(db_file)

    return experiment_id


def log_model_run(
    model_name: str,
    pipeline,
    params: dict,
    metrics: dict,
    dataset_info: dict,
    figures_dir: str = "reports/figures",
    feature_importance_plot: str = None
) -> str:
    """
    Log an individual model run to MLflow including params, metrics, dataset metadata,
    diagnostic plot artifacts, and the full pipeline model.

    Args:
        model_name (str): Name of the candidate model.
        pipeline: Scikit-learn Pipeline instance.
        params (dict): Hyperparameters for the model.
        metrics (dict): Performance metrics (MAE, MSE, RMSE, R2).
        dataset_info (dict): Dataset metadata (dataset_name, target_column, train_rows, test_rows, feature_count).
        figures_dir (str): Path to diagnostic figures directory.
        feature_importance_plot (str, optional): Path to feature importance plot file.

    Returns:
        str: MLflow Run ID.
    """
    with mlflow.start_run(run_name=model_name) as run:
        run_id = run.info.run_id

        # 1. Log Model Hyperparameters & Type
        mlflow.log_param("model_type", model_name)
        for key, val in params.items():
            mlflow.log_param(key, val)

        # 2. Log Dataset Metadata
        for tag_key, tag_val in dataset_info.items():
            mlflow.log_param(tag_key, tag_val)

        # 3. Log Performance Metrics
        for metric_name, val in metrics.items():
            mlflow.log_metric(metric_name, float(val))

        # 4. Log Diagnostic Plot Artifacts
        plot_files = [
            "actual_vs_predicted.png",
            "residual_plot.png",
            "prediction_error_distribution.png",
            "actual_vs_predicted_time.png"
        ]
        for plot_file in plot_files:
            plot_path = os.path.join(figures_dir, plot_file)
            if os.path.exists(plot_path):
                mlflow.log_artifact(plot_path, artifact_path="plots")

        if feature_importance_plot and os.path.exists(feature_importance_plot):
            mlflow.log_artifact(feature_importance_plot, artifact_path="plots")

        # 5. Log Pipeline Model Artifact with MLflow 2.x and 3.x compatibility
        skops_trusted = ["xgboost.core.Booster", "xgboost.sklearn.XGBRegressor"]
        try:
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                input_example=None,
                skops_trusted_types=skops_trusted
            )
        except TypeError:
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                input_example=None
            )

        return run_id


def register_best_model(
    run_id: str,
    model_name: str = "EnergyConsumptionModel",
    alias: str = "champion"
) -> dict:
    """
    Register the best model from MLflow run to the MLflow Model Registry,
    and set the specified alias (e.g. 'champion').

    Args:
        run_id (str): MLflow run ID of the best performing model.
        model_name (str): Name for registered model in registry.
        alias (str): Model version alias (e.g., 'champion').

    Returns:
        dict: Registry details including model name, version, run ID, and alias.
    """
    model_uri = f"runs:/{run_id}/model"
    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)

    client = MlflowClient()
    
    # Assign champion alias to the newly registered model version
    try:
        client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=registered_model.version
        )
    except Exception as e:
        print(f"[WARNING] MLflow alias assignment failed: {e}")

    registry_info = {
        "model_name": model_name,
        "version": registered_model.version,
        "run_id": run_id,
        "alias": alias
    }
    sanitize_mlflow_db_paths()
    return registry_info


def load_registered_model(model_name: str = "EnergyConsumptionModel", alias: str = "champion"):
    """
    Load registered model from MLflow Model Registry via alias or version.

    Args:
        model_name (str): Name of the registered model.
        alias (str): Model version alias (e.g. 'champion').

    Returns:
        Loaded PyFunc model instance.
    """
    setup_mlflow_experiment(experiment_name="Energy Consumption Prediction")
    model_uri = f"models:/{model_name}@{alias}"
    return mlflow.pyfunc.load_model(model_uri)
