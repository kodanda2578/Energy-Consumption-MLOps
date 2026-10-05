"""
Model Monitoring & Data Drift Detection Module for Energy Consumption Pipeline.

Implements Kolmogorov-Smirnov (KS) 2-sample statistical testing to compare incoming
feature distributions against reference baseline training data. Generates structured
monitoring reports, visualization charts, and logs tracking metrics to MLflow.
"""

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_config
from src.preprocessing import preprocess_data, get_feature_target_split, chronological_split
from src.mlflow_utils import setup_mlflow_experiment


def load_reference_data(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    """
    Load baseline reference training feature dataset.

    Args:
        config_path (str): Path to configuration YAML file.

    Returns:
        pd.DataFrame: Feature dataframe representing reference baseline training distribution.
    """
    config = load_config(config_path)
    processed_dir = config["data"]["processed_dir"]
    processed_filename = config["data"]["processed_filename"]
    processed_path = os.path.join(processed_dir, processed_filename)

    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path)
    else:
        df = preprocess_data(config_path)

    X, y, feature_names = get_feature_target_split(df, config_path)
    train_ratio = config.get("split", {}).get("train_ratio", 0.8)
    X_train, _, _, _ = chronological_split(X, y, train_ratio=train_ratio)

    return X_train


def detect_data_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    alpha: float = None,
    config_path: str = "configs/config.yaml"
) -> dict:
    """
    Perform Kolmogorov-Smirnov test to detect data drift between reference and current feature sets.

    Args:
        reference_df (pd.DataFrame): Baseline reference feature dataframe.
        current_df (pd.DataFrame): Current/incoming feature dataframe.
        alpha (float, optional): Significance threshold for p-value comparison. Defaults to config value (0.05).
        config_path (str): Path to configuration file.

    Returns:
        dict: Complete monitoring results including summary statistics and per-feature metrics.
    """
    config = load_config(config_path)
    if alpha is None:
        alpha = config.get("monitoring", {}).get("alpha", 0.05)

    # Identify common numerical columns
    ref_cols = set(reference_df.select_dtypes(include=[np.number]).columns)
    cur_cols = set(current_df.select_dtypes(include=[np.number]).columns)
    common_features = sorted(list(ref_cols.intersection(cur_cols)))

    if not common_features:
        raise ValueError("No common numerical features found between reference and current datasets.")

    feature_metrics = []
    drifted_count = 0

    for col in common_features:
        ref_series = reference_df[col].dropna()
        cur_series = current_df[col].dropna()

        if len(ref_series) == 0 or len(cur_series) == 0:
            continue

        res = ks_2samp(ref_series, cur_series)
        ks_stat = float(res.statistic)
        p_val = float(res.pvalue)
        is_drifted = bool(p_val < alpha)

        if is_drifted:
            drifted_count += 1

        feature_metrics.append({
            "feature": col,
            "reference_mean": float(ref_series.mean()),
            "reference_std": float(ref_series.std()),
            "current_mean": float(cur_series.mean()),
            "current_std": float(cur_series.std()),
            "ks_statistic": round(ks_stat, 6),
            "p_value": round(p_val, 6),
            "drift_detected": is_drifted
        })

    total_features = len(feature_metrics)
    drift_pct = round((drifted_count / total_features * 100.0), 2) if total_features > 0 else 0.0
    overall_drift = bool(drifted_count > 0)
    timestamp_str = datetime.datetime.now(datetime.timezone.utc).isoformat()

    summary = {
        "total_features": total_features,
        "drifted_features": drifted_count,
        "drift_percentage": drift_pct,
        "overall_drift_detected": overall_drift,
        "alpha": alpha,
        "timestamp": timestamp_str
    }

    return {
        "summary": summary,
        "features": feature_metrics
    }


def generate_monitoring_report(
    drift_results: dict,
    output_dir: str = None,
    config_path: str = "configs/config.yaml"
) -> dict:
    """
    Save monitoring results into human-readable JSON and CSV reports.

    Args:
        drift_results (dict): Output from detect_data_drift.
        output_dir (str, optional): Target directory for reports. Defaults to config setting.
        config_path (str): Path to configuration file.

    Returns:
        dict: Mapping of generated report file names to absolute paths.
    """
    if output_dir is None:
        config = load_config(config_path)
        output_dir = config.get("monitoring", {}).get("reports_dir", "reports/monitoring")

    os.makedirs(output_dir, exist_ok=True)

    summary_file = os.path.join(output_dir, "drift_summary.json")
    report_json_file = os.path.join(output_dir, "drift_report.json")
    report_csv_file = os.path.join(output_dir, "drift_report.csv")

    # 1. Save summary JSON
    with open(summary_file, "w") as f:
        json.dump(drift_results["summary"], f, indent=4)

    # 2. Save full report JSON
    with open(report_json_file, "w") as f:
        json.dump(drift_results, f, indent=4)

    # 3. Save feature breakdown CSV
    features_df = pd.DataFrame(drift_results["features"])
    features_df.to_csv(report_csv_file, index=False)

    return {
        "drift_summary": summary_file,
        "drift_report_json": report_json_file,
        "drift_report_csv": report_csv_file
    }


def plot_drift_metrics(
    drift_results: dict,
    output_dir: str = None,
    config_path: str = "configs/config.yaml"
) -> str:
    """
    Generate p-value visual chart showing feature drift comparison against threshold alpha.

    Args:
        drift_results (dict): Monitoring output from detect_data_drift.
        output_dir (str, optional): Target directory.
        config_path (str): Configuration file path.

    Returns:
        str: Absolute file path to saved plot image.
    """
    if output_dir is None:
        config = load_config(config_path)
        output_dir = config.get("monitoring", {}).get("reports_dir", "reports/monitoring")

    os.makedirs(output_dir, exist_ok=True)
    plot_file = os.path.join(output_dir, "drift_pvalues.png")

    features = drift_results["features"]
    alpha = drift_results["summary"]["alpha"]

    feat_names = [item["feature"] for item in features]
    p_values = [item["p_value"] for item in features]
    colors = ["#e74c3c" if item["drift_detected"] else "#2ecc71" for item in features]

    fig, ax = plt.subplots(figsize=(12, max(6, len(feat_names) * 0.35)))
    y_pos = np.arange(len(feat_names))

    ax.barh(y_pos, p_values, color=colors, alpha=0.85, edgecolor="black", height=0.6)
    ax.axvline(x=alpha, color="darkred", linestyle="--", linewidth=2, label=f"Alpha Threshold ({alpha})")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("KS Test p-value", fontsize=11, fontweight="bold")
    ax.set_title("Data Drift Feature p-values (Kolmogorov-Smirnov Test)", fontsize=13, fontweight="bold")
    ax.set_xlim(0.0, 1.05)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    plt.close(fig)

    return plot_file


def log_monitoring_to_mlflow(
    drift_results: dict,
    report_files: dict = None,
    figure_path: str = None,
    config_path: str = "configs/config.yaml"
) -> str:
    """
    Log data drift monitoring results, metrics, and report artifacts to MLflow tracking server.

    Args:
        drift_results (dict): Monitoring result dictionary.
        report_files (dict, optional): Paths to generated report files.
        figure_path (str, optional): Path to drift visualization plot.
        config_path (str): Path to config file.

    Returns:
        str: MLflow Run ID.
    """
    import mlflow

    setup_mlflow_experiment("Energy Consumption Prediction")

    with mlflow.start_run(run_name="data_drift_monitoring") as run:
        mlflow.set_tag("run_type", "data_drift_monitoring")
        mlflow.set_tag("model_name", "EnergyConsumptionModel")

        summary = drift_results["summary"]

        # Log parameters
        mlflow.log_param("alpha", summary["alpha"])
        mlflow.log_param("total_features_monitored", summary["total_features"])
        mlflow.log_param("monitoring_timestamp", summary["timestamp"])

        # Log metrics
        mlflow.log_metric("total_features", summary["total_features"])
        mlflow.log_metric("drifted_features", summary["drifted_features"])
        mlflow.log_metric("drift_percentage", summary["drift_percentage"])
        mlflow.log_metric("overall_drift_detected", 1 if summary["overall_drift_detected"] else 0)
        mlflow.log_metric("alpha", summary["alpha"])

        # Log per-feature metrics
        for feat in drift_results.get("features", []):
            safe_feat_name = feat["feature"].replace(" ", "_")
            mlflow.log_metric(f"p_val_{safe_feat_name}", feat["p_value"])
            mlflow.log_metric(f"ks_stat_{safe_feat_name}", feat["ks_statistic"])

        # Log report artifacts
        if report_files:
            for file_path in report_files.values():
                if os.path.exists(file_path):
                    mlflow.log_artifact(file_path, artifact_path="monitoring_reports")

        if figure_path and os.path.exists(figure_path):
            mlflow.log_artifact(figure_path, artifact_path="monitoring_reports")

        return run.info.run_id


def run_monitoring_pipeline(
    current_df: pd.DataFrame = None,
    alpha: float = None,
    config_path: str = "configs/config.yaml",
    log_to_mlflow: bool = True
) -> dict:
    """
    Execute complete end-to-end data drift monitoring workflow.

    Args:
        current_df (pd.DataFrame, optional): Incoming dataset to monitor. Defaults to baseline reference data.
        alpha (float, optional): Significance threshold.
        config_path (str): Path to configuration YAML.
        log_to_mlflow (bool): Whether to log monitoring run to MLflow.

    Returns:
        dict: Complete monitoring results.
    """
    reference_df = load_reference_data(config_path)

    if current_df is None:
        current_df = reference_df.copy()

    drift_results = detect_data_drift(reference_df, current_df, alpha=alpha, config_path=config_path)
    report_files = generate_monitoring_report(drift_results, config_path=config_path)
    figure_path = plot_drift_metrics(drift_results, config_path=config_path)

    if log_to_mlflow:
        run_id = log_monitoring_to_mlflow(
            drift_results=drift_results,
            report_files=report_files,
            figure_path=figure_path,
            config_path=config_path
        )
        drift_results["mlflow_run_id"] = run_id

    return drift_results


if __name__ == "__main__":
    print("[INFO] Running standalone data drift monitoring pipeline...")
    results = run_monitoring_pipeline()
    print(f"[SUCCESS] Monitoring completed. Drift Summary: {json.dumps(results['summary'], indent=2)}")
