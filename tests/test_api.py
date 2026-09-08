"""
Unit & Integration tests for FastAPI model serving application.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app, model_state


@pytest.fixture
def client():
    """
    TestClient context manager fixture that triggers FastAPI lifespan startup/shutdown events.
    """
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    """
    Verify GET / returns 200 OK and valid application info.
    """
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "project" in data
    assert data["status"] == "online"
    assert "docs_url" in data


def test_health_endpoint(client):
    """
    Verify GET /health returns 200 OK and model readiness status.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_name"] == "EnergyConsumptionModel"
    assert data["model_alias"] == "champion"


def test_predict_endpoint_valid_input(client):
    """
    Verify POST /predict with a valid sample payload returns 200 OK and numeric prediction.
    """
    sample_payload = {
        "lights": 0,
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

    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_consumption" in data
    assert isinstance(data["predicted_consumption"], (int, float))
    assert data["model_name"] == "EnergyConsumptionModel"
    assert data["model_alias"] == "champion"


def test_predict_endpoint_invalid_data_type(client):
    """
    Verify POST /predict with invalid data type returns 422 Unprocessable Entity.
    """
    invalid_payload = {
        "lights": "invalid_string_type",
        "T1": 19.89
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_endpoint_missing_required_fields(client):
    """
    Verify POST /predict with missing required fields returns 422 Unprocessable Entity.
    """
    incomplete_payload = {
        "lights": 10,
        "T1": 19.89
        # Missing all remaining required features
    }
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422


def test_model_unloaded_state_handling():
    """
    Verify that if the model is not loaded, /health reports unhealthy and /predict returns 503.
    """
    with TestClient(app) as client:
        # Override state inside TestClient block after lifespan runs
        original_loaded = model_state["model_loaded"]
        original_model = model_state["model"]

        model_state["model_loaded"] = False
        model_state["model"] = None

        try:
            health_resp = client.get("/health")
            assert health_resp.status_code == 200
            assert health_resp.json()["status"] == "unhealthy"
            assert health_resp.json()["model_loaded"] is False

            valid_sample_payload = {
                "lights": 0,
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
            pred_resp = client.post("/predict", json=valid_sample_payload)
            assert pred_resp.status_code == 503
        finally:
            # Restore original model state
            model_state["model_loaded"] = original_loaded
            model_state["model"] = original_model


