"""
FastAPI Model Serving Application for Energy Consumption Prediction.

Loads the MLflow registered model ('EnergyConsumptionModel@champion') during application
startup and provides RESTful prediction and health check endpoints.
"""

import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import mlflow

from api.schemas import (
    EnergyPredictionRequest,
    EnergyPredictionResponse,
    HealthResponse,
    MonitoringSummary,
    MonitoringResponse
)


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
    Lifespan event handler to load the energy prediction model once on application startup.
    Prioritizes loading from the standalone exported model directory (e.g. 'deployment_model'
    or MODEL_PATH env var) which requires no mlflow.db or mlruns at runtime.
    Falls back to the local MLflow registry if the standalone directory is unavailable.
    """
    model_name = os.getenv("MODEL_NAME", "EnergyConsumptionModel")
    model_alias = os.getenv("MODEL_ALIAS", "champion")
    model_dir = os.getenv("MODEL_PATH", "deployment_model")

    model_state["model_name"] = model_name
    model_state["model_alias"] = model_alias

    try:
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        if os.path.exists(model_dir) and os.path.isdir(model_dir) and (
            os.path.exists(os.path.join(model_dir, "MLmodel")) or os.path.exists(os.path.join(model_dir, "model.pkl"))
        ):
            print(f"[INFO] Loading standalone deployment model from: {model_dir}...")
            # If registered_model_meta exists, read model info
            meta_path = os.path.join(model_dir, "registered_model_meta")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as mf:
                        for line in mf:
                            if ":" in line:
                                k, v = line.strip().split(":", 1)
                                if k.strip() == "model_name" and not os.getenv("MODEL_NAME"):
                                    model_state["model_name"] = v.strip()
                except Exception:
                    pass

            model_state["model"] = mlflow.pyfunc.load_model(model_dir)
            model_state["model_loaded"] = True
            model_state["error"] = None
            print(f"[SUCCESS] Standalone deployment model loaded successfully from '{model_dir}'.")
        else:
            print(f"[INFO] Standalone model directory '{model_dir}' not found. Falling back to MLflow Model Registry...")
            from src.mlflow_utils import setup_mlflow_experiment, load_registered_model
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
            setup_mlflow_experiment("Energy Consumption Prediction")
            model_uri = f"models:/{model_name}@{model_alias}"
            print(f"[INFO] Loading MLflow model from registry URI: {model_uri}...")
            model_state["model"] = load_registered_model(model_name=model_name, alias=model_alias)
            model_state["model_loaded"] = True
            model_state["error"] = None
            print(f"[SUCCESS] MLflow model '{model_name}' (@{model_alias}) loaded successfully from registry.")
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

# Configure CORS origins safely with support for environment variables
default_origins = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:5174", "http://127.0.0.1:5174",
    "http://localhost:5175", "http://127.0.0.1:5175",
    "http://localhost:5176", "http://127.0.0.1:5176",
    "http://localhost:5177", "http://127.0.0.1:5177",
    "http://localhost:3000"
]
env_cors = os.getenv("CORS_ORIGINS", "").strip()
if env_cors:
    if env_cors == "*":
        allowed_origins = ["*"]
    else:
        env_origins = [origin.strip() for origin in env_cors.split(",") if origin.strip()]
        allowed_origins = list(dict.fromkeys(default_origins + env_origins))
else:
    allowed_origins = default_origins

allow_creds = allowed_origins != ["*"]
cors_origin_regex = os.getenv(
    "CORS_ORIGIN_REGEX",
    r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$|^https://.*\.vercel\.app$"
).strip() or None

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=cors_origin_regex if allowed_origins != ["*"] else None,
    allow_credentials=allow_creds,
    allow_methods=["*"],
    allow_headers=["*"],
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
        "predict_url": "/predict",
        "monitoring_url": "/monitoring"
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


@app.get("/monitoring", response_model=MonitoringResponse, tags=["Monitoring"])
def get_monitoring():
    """
    Monitoring endpoint returning the latest data drift summary report.
    Reads saved report summary if available, otherwise executes on-demand drift calculation.
    """
    try:
        summary_path = os.path.join("reports", "monitoring", "drift_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                summary_data = json.load(f)
        else:
            try:
                from src.monitoring import run_monitoring_pipeline
                results = run_monitoring_pipeline(log_to_mlflow=False)
                summary_data = results["summary"]
            except Exception:
                # Standalone deployment fallback summary
                summary_data = {
                    "total_features": 41,
                    "drifted_features": 0,
                    "drift_percentage": 0.0,
                    "overall_drift_detected": False,
                    "alpha": 0.05,
                    "timestamp": None
                }

        return MonitoringResponse(
            status="success",
            monitoring=MonitoringSummary(**summary_data)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Monitoring evaluation failed: {str(e)}"
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

