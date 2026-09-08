"""
FastAPI Model Serving Application for Energy Consumption Prediction.

Loads the MLflow registered model ('EnergyConsumptionModel@champion') during application
startup and provides RESTful prediction and health check endpoints.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
import pandas as pd
import mlflow

from api.schemas import EnergyPredictionRequest, EnergyPredictionResponse, HealthResponse
from src.mlflow_utils import setup_mlflow_experiment


# Global dictionary holding application model state
model_state = {
    "model": None,
    "model_loaded": False,
    "model_name": "EnergyConsumptionModel",
    "model_alias": "champion",
    "error": None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler to load the MLflow champion model once on application startup.
    """
    try:
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        setup_mlflow_experiment("Energy Consumption Prediction", tracking_uri="sqlite:///mlflow.db")
        model_uri = f"models:/{model_state['model_name']}@{model_state['model_alias']}"
        print(f"[INFO] Loading MLflow model from registry URI: {model_uri}...")
        model_state["model"] = mlflow.pyfunc.load_model(model_uri)
        model_state["model_loaded"] = True
        model_state["error"] = None
        print(f"[SUCCESS] MLflow model '{model_state['model_name']}' (@{model_state['model_alias']}) loaded successfully.")
    except Exception as e:
        model_state["model"] = None
        model_state["model_loaded"] = False
        model_state["error"] = str(e)
        print(f"[ERROR] Model loading failed: {e}")

    yield

    # Cleanup state on shutdown
    model_state["model"] = None
    model_state["model_loaded"] = False


app = FastAPI(
    title="Energy Consumption Prediction API",
    description="Production MLOps FastAPI service predicting appliance energy consumption using MLflow registered models.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/", tags=["Root"])
def read_root():
    """
    Root endpoint displaying project overview and interactive API documentation links.
    """
    return {
        "project": "Energy Consumption Prediction with MLOps",
        "status": "online",
        "description": "FastAPI REST service serving MLflow registered energy prediction model.",
        "docs_url": "/docs",
        "health_url": "/health",
        "predict_url": "/predict"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health Check"])
def health_check():
    """
    Health check endpoint returning system operational status and model readiness.
    """
    is_healthy = model_state["model_loaded"]
    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        model_loaded=model_state["model_loaded"],
        model_name=model_state["model_name"],
        model_alias=model_state["model_alias"]
    )


@app.post("/predict", response_model=EnergyPredictionResponse, tags=["Prediction"])
def predict_energy_consumption(payload: EnergyPredictionRequest):
    """
    Prediction endpoint accepting feature payloads and returning energy consumption forecasts.
    """
    if not model_state["model_loaded"] or model_state["model"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model is not loaded or unavailable: {model_state.get('error', 'Unknown error')}"
        )

    try:
        # Convert input Pydantic payload to single-row pandas DataFrame
        input_data = payload.model_dump()
        input_df = pd.DataFrame([input_data])

        # Run prediction through loaded PyFunc MLflow pipeline
        predictions = model_state["model"].predict(input_df)
        predicted_val = float(predictions[0])

        return EnergyPredictionResponse(
            predicted_consumption=round(predicted_val, 2),
            model_name=model_state["model_name"],
            model_alias=model_state["model_alias"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )
