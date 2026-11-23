# Energy Consumption Prediction with MLOps

## 📌 Project Objective
Build an end-to-end, production-grade Machine Learning system that predicts appliance energy consumption based on historical, time, weather, and indoor climate features. The project follows software engineering and MLOps best practices—spanning data preprocessing, experiment tracking, model registry, standalone model export, RESTful API serving, containerization, CI/CD automation, interactive user interface, and continuous data drift monitoring.

---

## 📊 Selected Dataset
* **Dataset Name:** Appliances Energy Prediction Dataset
* **Source:** UCI Machine Learning Repository (ID: 374)
* **Dataset URL:** [UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)
* **Records / Features:** 19,735 observations, 29 raw attributes (sampled every 10 minutes over 4.5 months)
* **Target Variable:** `Appliances` (Appliance energy use in Wh)
* **Key Feature Categories:**
  * **Temporal Features:** `date` (Timestamp at 10-minute intervals)
  * **Indoor Microclimate:** Temperatures (`T1`–`T9`) & Relative Humidity (`RH_1`–`RH_9`) across 9 rooms
  * **Outdoor Weather Data:** `T_out` (Temperature), `Press_mm_hg` (Pressure), `RH_out` (Humidity), `Windspeed`, `Visibility`, `Tdewpoint`

---

## 🔒 Data Leakage Prevention Strategy
To ensure real-world validity in predicting time-dependent energy demand:
1. **Chronological Train/Test Split:** Data is split strictly by time (first 80% training, final 20% testing). No random shuffling is used to prevent future information from leaking into training folds.
2. **Shifted Lags & Rolling Window Statistics:** All target lag features (`appliances_lag_1`, `appliances_lag_3`, etc.) and rolling statistics (`rolling_mean_3`, `rolling_mean_6`, etc.) use `.shift(1)` before window computation. This guarantees that prediction at time $t$ relies strictly on past historical values ($t-1, t-2, \dots$) and never includes $t$.
3. **Training-Only Preprocessing Scaling:** Scalers (`StandardScaler`) are fitted strictly on the training set (`X_train`) inside a `sklearn.pipeline.Pipeline`, preventing target/test distribution leakage.

---

## 📈 Verified Model Performance Results

| Model | MAE (Wh) | RMSE (Wh) | R² Score | Status |
| :--- | ---: | ---: | ---: | :---: |
| **Linear Regression** ⭐ | **27.63** | **59.77** | **0.5650** | **Champion Model (Version 30)** |
| **XGBoost Regressor** | 50.36 | 78.26 | 0.2541 | Candidate |
| **Random Forest Regressor** | 69.23 | 103.39 | -0.3018 | Candidate |

*Linear Regression achieved the lowest RMSE (59.77) and highest R² (0.5650) on the chronological test set.*
* **MLflow Registered Model:** `EnergyConsumptionModel`
* **Verified Model Version:** `30` (Alias: `@champion`)
* **Run ID:** `f873c17ccb1f4ee3b06e27ae1872b575`
* **Features:** 41 engineered features

---

## 🏗️ Production Architecture

```
                                 +--------------------+
                                 |    Raw Data        | (UCI Energy Dataset)
                                 +---------+----------+
                                           |
                                           v
                                 +---------+----------+
                                 |   Data Pipeline    | (Pandas / DVC)
                                 +---------+----------+
                                           |
                                           v
                                 +---------+----------+
                                 |   Model Training   | (Scikit-Learn / XGBoost)
                                 +---------+----------+
                                           |
                                           v
                                 +---------+----------+
                                 |Experiment Tracking | (MLflow / SQLite)
                                 +---------+----------+
                                           |
                                           v
                                 +---------+----------+
                                 |   Model Registry   | (models:/EnergyConsumptionModel/30)
                                 +---------+----------+
                                           |
                                           v  [src/export_model.py]
                                 +---------+----------+
                                 | Standalone Model   | (deployment_model/)
                                 |  (Self-Contained)  | (No DB or mlruns required)
                                 +---------+----------+
                                           |
                                           v
                                 +---------+----------+
                                 |   FastAPI Server   | (Render Web Service / Docker)
                                 +----+----------+----+
                                      |          |
                     +----------------+          +----------------+
                     v                                            v
          +----------+----------+                      +----------+----------+
          | React + Vite App    |                      | Data Drift & Health |
          | (Deployed on Vercel)|                      | Monitoring Pipeline |
          +---------------------+                      +---------------------+
```

---

## 🛠️ Technology Stack

* **Programming Language:** Python 3.11+
* **Machine Learning & Data Processing:** Pandas, NumPy, Scikit-Learn, XGBoost, Joblib
* **Experiment Tracking & Model Registry:** MLflow 3.17+
* **Backend REST API:** FastAPI, Uvicorn, Pydantic v2
* **Frontend Dashboard:** React 18, Vite, Lucide React, Recharts
* **Containerization:** Docker, Docker Compose
* **Testing & Quality Assurance:** Pytest, Fastapi TestClient, HTTPX
* **Statistical Monitoring:** SciPy (Kolmogorov-Smirnov two-sample test)
* **Cloud Hosting:** Render (Backend API), Vercel (Frontend Dashboard)

---

## 📦 Standalone Model Export (Production Serving)

The verified champion model (Version 30) is exported into a self-contained, standalone directory `deployment_model/` using the official MLflow artifacts API:

```bash
python src/export_model.py --model-name EnergyConsumptionModel --version 30 --output-dir deployment_model
```

### Advantages of Standalone Export:
1. **Zero Database Dependency:** FastAPI loads the exported pipeline directly via `mlflow.pyfunc.load_model("deployment_model")` without requiring `mlflow.db` (SQLite) or the multi-gigabyte `mlruns/` directory in production.
2. **Lightweight & Portable:** The entire `deployment_model/` folder is < 5 KB total (`model.pkl`, `MLmodel`, `conda.yaml`, `requirements.txt`).
3. **Graceful Fallback:** If `deployment_model/` is absent, FastAPI automatically falls back to loading from the local MLflow registry via SQLite URI.

---

## 🌐 API Endpoints Reference

The FastAPI service exposes four primary endpoints:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Root info: project name, status, and API documentation links |
| `GET` | `/health` | Service health status and loaded model readiness |
| `POST` | `/predict` | Real-time appliance energy consumption forecast (requires 41 features) |
| `GET` | `/monitoring` | Data drift detection summary (Kolmogorov-Smirnov test results) |
| `GET` | `/docs` | Interactive Swagger UI documentation |

### Sample Prediction Request (`POST /predict`):
```json
{
  "lights": 0, "T1": 19.89, "RH_1": 47.59, "T2": 19.2, "RH_2": 44.79,
  "T3": 19.79, "RH_3": 44.73, "T4": 19.0, "RH_4": 45.56, "T5": 17.1667,
  "RH_5": 55.2, "T6": 7.0267, "RH_6": 84.2567, "T7": 17.2, "RH_7": 41.6267,
  "T8": 18.2, "RH_8": 48.9, "T9": 17.0333, "RH_9": 45.53, "T_out": 6.6,
  "Press_mm_hg": 733.5, "RH_out": 92.0, "Windspeed": 7.0, "Visibility": 63.0,
  "Tdewpoint": 5.3, "hour": 18, "day_of_week": 0, "month": 1, "day": 11,
  "is_weekend": 0, "sin_hour": -1.0, "cos_hour": 0.0, "sin_day_of_week": 0.0,
  "cos_day_of_week": 1.0, "appliances_lag_1": 60.0, "appliances_lag_3": 50.0,
  "appliances_lag_6": 50.0, "appliances_lag_12": 60.0, "rolling_mean_3": 53.3333,
  "rolling_mean_6": 55.0, "rolling_mean_12": 58.3333
}
```

### Sample Prediction Response:
```json
{
  "predicted_consumption": 64.01,
  "model_name": "EnergyConsumptionModel",
  "model_alias": "champion"
}
```

---

## 💻 Local Setup & Execution Guide

### 1. Prerequisites
* Python 3.11+
* Node.js v18+ and npm v9+
* Git

### 2. Backend Setup
```bash
# Clone the repository
git clone <your-repository-url>
cd Energy-Consumption-MLOps

# Create & activate virtual environment
python -m venv .venv
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On macOS / Linux:
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
* Interactive Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
* Health Check: [http://localhost:8000/health](http://localhost:8000/health)

### 3. Frontend Setup
```bash
# In a new terminal window
cd frontend

# Install Node dependencies
npm install

# Start development server
npm run dev
```
* Dashboard URL: [http://localhost:5173](http://localhost:5173) (or [http://localhost:5175](http://localhost:5175))

### 4. Running Backend Tests & Frontend Build
```bash
# Run backend test suite
pytest tests/test_api.py tests/test_data_loader.py tests/test_preprocessing.py tests/test_monitoring.py -v

# Run production frontend build
cd frontend
npm run build
```

---

## 🚀 Deployment Guide

### A. Backend Deployment on Render

Render can deploy the FastAPI service using either **Render Blueprints (`render.yaml`)** or a **Web Service**.

#### Option 1: Blueprint Deployment (Recommended)
1. Push your repository to GitHub.
2. Go to [Render Dashboard](https://dashboard.render.com/) -> **New +** -> **Blueprint**.
3. Select your repository. Render automatically reads `render.yaml`.
4. Click **Apply**. Render will automatically provision the service, install dependencies, and start Uvicorn.

#### Option 2: Manual Web Service Deployment
1. Go to **New +** -> **Web Service**.
2. Select your repository.
3. Configure the service:
   * **Name:** `energy-consumption-mlops-api`
   * **Language:** `Python 3`
   * **Region:** Any (e.g. `Oregon`)
   * **Branch:** `master` / `main`
   * **Build Command:** `pip install --no-cache-dir -r requirements.txt`
   * **Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
4. Add **Environment Variables** in the Render Dashboard:
   | Variable | Value | Description |
   | :--- | :--- | :--- |
   | `PYTHON_VERSION` | `3.11.9` | Sets the Python version |
   | `MODEL_PATH` | `deployment_model` | Points to exported standalone model |
   | `MODEL_NAME` | `EnergyConsumptionModel` | Model registry name |
   | `MODEL_ALIAS` | `champion` | Model version alias |
   | `MLFLOW_DISABLE_AGENT_HINT` | `1` | Suppresses MLflow CLI warnings |
   | `CORS_ORIGIN_REGEX` | `^https?://(localhost\|127\.0\.0\.1)(:\d+)?$\|^https://.*\.vercel\.app$` | Allows localhost & all Vercel deployments |
   | `CORS_ORIGINS` | `https://your-frontend-app.vercel.app` | Explicit frontend domain (optional) |
5. Click **Create Web Service**.
6. Once deployed, test the health endpoint: `https://<your-render-url>.onrender.com/health`.

---

### B. Frontend Deployment on Vercel

The React frontend is configured for instant deployment on Vercel using `frontend/vercel.json`.

#### Deployment Steps:
1. Go to [Vercel Dashboard](https://vercel.com/) -> **Add New...** -> **Project**.
2. Import your GitHub repository.
3. In the configuration screen:
   * **Framework Preset:** `Vite`
   * **Root Directory:** Click **Edit** and select `frontend`
   * **Build Command:** `npm run build` (automatic)
   * **Output Directory:** `dist` (automatic)
4. Add **Environment Variable**:
   * **Name:** `VITE_API_BASE_URL`
   * **Value:** `https://<your-render-app>.onrender.com` (Your deployed Render backend URL, without trailing slash)
5. Click **Deploy**.
6. Vercel builds the SPA, configures the single-page application routing rewrites via `vercel.json`, and provisions a global HTTPS URL (`https://<project-name>.vercel.app`).

---

## ⚙️ Environment Variables Reference

### Backend (`.env` or Cloud Provider Settings)
| Variable | Default Value | Purpose |
| :--- | :--- | :--- |
| `MODEL_PATH` | `deployment_model` | Path to standalone exported model directory |
| `MODEL_NAME` | `EnergyConsumptionModel` | Name of the registered model |
| `MODEL_ALIAS` | `champion` | Alias of champion model version |
| `CORS_ORIGINS` | *(empty)* | Comma-separated list of allowed origins or `*` |
| `CORS_ORIGIN_REGEX` | `^https?://(localhost\|127\.0\.0\.1)(:\d+)?$\|^https://.*\.vercel\.app$` | Regex pattern matching allowed origins |
| `MLFLOW_DISABLE_AGENT_HINT` | `1` | Disables MLflow tracing hints in console |
| `PORT` | `8000` | Port for Uvicorn server (auto-assigned by Render) |

### Frontend (`frontend/.env` or Vercel Environment Variables)
| Variable | Default Value | Purpose |
| :--- | :--- | :--- |
| `VITE_API_BASE_URL` | `http://localhost:8000` | Base URL of the FastAPI backend service |

---

## 🎨 React + Vite MLOps Dashboard

### Key Dashboard Views
- **Dashboard View**: Real-time KPI summary cards (Model status, Champion version, Version alias, Data drift alert).
- **Interactive Prediction UI**: 41-feature input form categorized into Indoor Climate, Outdoor Weather, Calendar/Time, Lags, and Rolling Statistics, with instant 1-click benchmark preset loading.
- **Data Drift Monitoring View**: Statistical Kolmogorov-Smirnov ($\alpha = 0.05$) metric cards, stability ratios, and drifted feature detection.
- **Model Information View**: Comparative performance metrics (RMSE: 59.77 Wh, MAE: 27.63 Wh, R²: 0.5650) with interactive Recharts bar charts.
- **System Health Diagnostics**: Live API readiness monitor with real-time connectivity testing across `/`, `/health`, `/predict`, and `/monitoring`.

---

## 🔍 Model Monitoring & Data Drift Architecture

1. **Baseline Reference Dataset:** Preprocessed training features (`X_train` from `data/processed/energy_consumption_processed.csv`).
2. **Current Telemetry:** Incoming inference features or recent batches.
3. **Statistical Test:** Two-sample Kolmogorov-Smirnov (KS) test (`scipy.stats.ks_2samp`) on all 41 features.
4. **Significance Threshold:** Configurable $\alpha = 0.05$. A feature is flagged as drifted if $p\text{-value} < \alpha$.
5. **FastAPI Endpoint:** `GET /monitoring` returns the current summary report without requiring model re-execution.
