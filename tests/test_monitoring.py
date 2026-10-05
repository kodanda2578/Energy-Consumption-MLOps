"""
Unit & Integration tests for Model Monitoring and Data Drift Detection module.
"""

import os
import json
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from src.monitoring import (
    load_reference_data,
    detect_data_drift,
    generate_monitoring_report,
    plot_drift_metrics,
    log_monitoring_to_mlflow,
    run_monitoring_pipeline
)
from api.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_monitoring_imports():
    """
    1. Verify monitoring module imports correctly.
    """
    assert callable(load_reference_data)
    assert callable(detect_data_drift)
    assert callable(generate_monitoring_report)
    assert callable(plot_drift_metrics)
    assert callable(log_monitoring_to_mlflow)
    assert callable(run_monitoring_pipeline)


def test_reference_data_loading():
    """
    2. Verify baseline reference dataset loads correctly and contains numerical features.
    """
    ref_df = load_reference_data()
    assert isinstance(ref_df, pd.DataFrame)
    assert not ref_df.empty
    assert ref_df.shape[1] > 10


def test_identical_data_produces_no_drift():
    """
    3. Verify identical reference and current datasets yield 0 drifted features.
    """
    ref_df = load_reference_data()
    current_df = ref_df.copy()

    results = detect_data_drift(ref_df, current_df, alpha=0.05)
    summary = results["summary"]

    assert summary["drifted_features"] == 0
    assert summary["drift_percentage"] == 0.0
    assert summary["overall_drift_detected"] is False


def test_shifted_data_detects_drift():
    """
    4. Verify artificially shifted feature data triggers data drift detection.
    """
    ref_df = load_reference_data()
    shifted_df = ref_df.copy()
    
    # Artificially shift values for all columns by adding noise/scaling
    for col in shifted_df.select_dtypes(include=[np.number]).columns:
        shifted_df[col] = shifted_df[col] * 2.5 + 50.0

    results = detect_data_drift(ref_df, shifted_df, alpha=0.05)
    summary = results["summary"]

    assert summary["drifted_features"] > 0
    assert summary["drift_percentage"] > 0.0
    assert summary["overall_drift_detected"] is True


def test_drift_summary_calculation():
    """
    5. Verify summary calculations (total features, drift percentage, keys).
    """
    ref_df = load_reference_data()
    results = detect_data_drift(ref_df, ref_df, alpha=0.05)
    summary = results["summary"]

    expected_keys = [
        "total_features", "drifted_features", "drift_percentage",
        "overall_drift_detected", "alpha", "timestamp"
    ]
    for key in expected_keys:
        assert key in summary

    assert summary["total_features"] == len(results["features"])
    assert isinstance(summary["drift_percentage"], float)


def test_alpha_threshold_respected():
    """
    6. Verify that changing alpha alters drift sensitivity threshold.
    """
    ref_df = load_reference_data()
    slightly_shifted_df = ref_df.copy()
    num_cols = list(slightly_shifted_df.select_dtypes(include=[np.number]).columns)
    
    # Introduce small shift to first few columns
    for col in num_cols[:3]:
        slightly_shifted_df[col] = slightly_shifted_df[col] + 0.5 * slightly_shifted_df[col].std()

    strict_res = detect_data_drift(ref_df, slightly_shifted_df, alpha=0.99)
    lenient_res = detect_data_drift(ref_df, slightly_shifted_df, alpha=1e-15)

    assert strict_res["summary"]["alpha"] == 0.99
    assert lenient_res["summary"]["alpha"] == 1e-15
    assert strict_res["summary"]["drifted_features"] >= lenient_res["summary"]["drifted_features"]


def test_monitoring_report_generation(tmp_path):
    """
    7. Verify monitoring reports (JSON & CSV) and plots are written to disk.
    """
    ref_df = load_reference_data()
    results = detect_data_drift(ref_df, ref_df, alpha=0.05)
    
    report_files = generate_monitoring_report(results, output_dir=str(tmp_path))
    plot_file = plot_drift_metrics(results, output_dir=str(tmp_path))

    assert os.path.exists(report_files["drift_summary"])
    assert os.path.exists(report_files["drift_report_json"])
    assert os.path.exists(report_files["drift_report_csv"])
    assert os.path.exists(plot_file)

    with open(report_files["drift_summary"], "r") as f:
        summary_data = json.load(f)
        assert summary_data["total_features"] == results["summary"]["total_features"]


def test_mlflow_monitoring_run_creation():
    """
    8. Verify logging data drift results creates an MLflow run with run_type tag.
    """
    ref_df = load_reference_data()
    results = detect_data_drift(ref_df, ref_df, alpha=0.05)
    run_id = log_monitoring_to_mlflow(results)

    assert run_id is not None
    assert len(run_id) > 0


def test_fastapi_monitoring_endpoint(client):
    """
    9. Verify GET /monitoring returns 200 OK and valid monitoring response structure.
    """
    response = client.get("/monitoring")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert "monitoring" in data
    mon = data["monitoring"]
    assert "total_features" in mon
    assert "drifted_features" in mon
    assert "drift_percentage" in mon
    assert "overall_drift_detected" in mon
    assert "alpha" in mon
