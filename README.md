# Energy Consumption Prediction with MLOps

## 📌 Project Objective
Build an end-to-end, production-style Machine Learning system that predicts energy consumption based on historical, time, weather, and energy usage features. The project follows software engineering and MLOps best practices—spanning data preprocessing, experiment tracking, model registry, API deployment, containerization, CI/CD automation, interactive user interface, and system monitoring.

---

## 📊 Selected Dataset
* **Dataset Name:** Appliances Energy Prediction Dataset
* **Source:** UCI Machine Learning Repository (ID: 374)
* **Dataset URL:** [UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)
* **Records / Features:** 19,735 observations, 29 attributes (sampled every 10 minutes over 4.5 months)
* **Target Variable:** `Appliances` (Appliance energy use in Wh)
* **Key Feature Categories:**
  * **Temporal Features:** `date` (Timestamp at 10-minute intervals)
  * **Indoor Microclimate:** Temperatures (`T1`–`T9`) & Relative Humidity (`RH_1`–`RH_9`) across 9 rooms
  * **Outdoor Weather Data:** `T_out` (Temperature), `Press_mm_hg` (Pressure), `RH_out` (Humidity), `Windspeed`, `Visibility`, `Tdewpoint`

---

## 🏗️ Planned Architecture

```
                                +-------------------+
                                |    Raw Data       | (UCI Energy Dataset)
                                +---------+---------+
                                          |
                                          v
                                +---------+---------+
                                |  Data Pipeline    | (Pandas / DVC)
                                +---------+---------+
                                          |
                                          v
                                +---------+---------+
                                |  Model Training   | (Scikit-Learn / XGBoost)
                                +---------+---------+
                                          |
                                          v
                                +---------+---------+
                                |Experiment Tracking| (MLflow)
                                +---------+---------+
                                          |
                                          v
                                +---------+---------+
                                |   Model Registry  |
                                +---------+---------+
                                          |
                                          v
                                +---------+---------+
                                |   FastAPI Server  | (Docker Container)
                                +----+---------+----+
                                     |         |
                    +----------------+         +----------------+
                    v                                           v
         +----------+----------+                     +----------+----------+
         | Streamlit Web App   |                     | Prometheus & Grafana |
         |   (User Interface)  |                     | (System Monitoring)  |
         +---------------------+                     +---------------------+
```

---

## 🛠️ Technology Stack

* **Programming Language:** Python 3.10+
* **Machine Learning & Data Processing:** Pandas, NumPy, Scikit-Learn, XGBoost
* **Data Visualization:** Matplotlib, Seaborn
* **Data & Model Versioning:** Git, DVC, MLflow
* **API & Serving:** FastAPI, Uvicorn
* **Frontend UI:** Streamlit
* **Containerization & CI/CD:** Docker, GitHub Actions
* **Testing & Quality:** Pytest
* **Monitoring:** Prometheus & Grafana (lightweight monitoring setup)

---

## 🗺️ Development Roadmap

- [x] **Phase 1: Project Setup & Structure**
  - Modular directory design & VS Code configuration
  - Dependency definition (`requirements.txt`) & configuration template (`config.yaml`)
- [x] **Phase 2: Data Intake & Exploratory Data Analysis (EDA)** (Current)
  - Dataset evaluation & selection (UCI Appliance Energy Dataset)
  - Automated reproducible data loader module (`src/data_loader.py`)
  - Exploratory analysis notebook (`notebooks/01_exploratory_data_analysis.ipynb`)
- [x] **Phase 3: Data Preprocessing & Model Training Baseline**
  - Feature engineering (lags, rolling averages, temporal components)
  - Chronological data split, baseline regression models (Linear Regression, Random Forest, XGBoost)
  - Evaluation metrics (RMSE, MAE, R²) and pipeline serialization
- [x] **Phase 4: MLOps Integration (Experiment Tracking with MLflow & Versioning)**
  - Tracking hyperparameters, metrics (MAE, MSE, RMSE, R²), and artifacts with MLflow
  - Model Registry (`EnergyConsumptionModel`) and champion version alias tagging (`champion`)
- [x] **Phase 5: Data Version Control using DVC**
  - Tracking raw dataset (`data/raw/energydata_complete.csv`) using DVC
  - Local DVC remote storage (`dvc_storage/`) and checkout/pull reproducibility verification
- [x] **Phase 6: API Development with FastAPI**
  - Production RESTful prediction API (`api/main.py`)
  - MLflow `champion` model loading at application startup
  - Endpoints: `GET /`, `GET /health`, `POST /predict`, `/docs` (Swagger UI)
- [x] **Phase 7: Docker Containerization**
  - Lightweight production container (`Dockerfile`, `.dockerignore`, `docker-compose.yml`)
  - Container health checks and single-command orchestration
- [x] **Phase 8: GitHub Actions CI/CD Workflow**
  - Automated testing and Docker build verification on push/pull_request to master (`.github/workflows/ci.yml`)



---

## 📦 Data Version Control with DVC

Data Version Control (DVC) is integrated into the repository to manage dataset versioning and ensure total data reproducibility without committing large raw data files into Git.

### Why Git Alone is Not Ideal for Large Datasets
Git is designed for text-based source code files. Storing large binary datasets directly in Git leads to:
* Repository bloat and degraded performance.
* Inefficient diffing and merge conflicts on large files.
* Storage bandwidth waste on remote git providers (e.g. GitHub).

### What DVC Does & How Datasets are Tracked
* **Metadata Tracking:** DVC hashes data files (using MD5 checksums) and generates lightweight pointer files with a `.dvc` extension (e.g. `data/raw/energydata_complete.csv.dvc`).
* **Git Integration:** Only the lightweight `.dvc` pointer files and `.dvc/config` are committed to Git. The actual heavy data files remain ignored by Git via `data/raw/.gitignore`.
* **Local DVC Remote:** Actual raw data payloads are stored in a dedicated local DVC storage directory (`dvc_storage/`) acting as the local storage remote.

### DVC vs. MLflow Responsibilities
| Responsibility | Managed By | Description |
| :--- | :--- | :--- |
| **Dataset Versioning** | **DVC** | Tracks exact dataset versions, raw file hashes, and dataset storage remotes (`.dvc` pointer files). |
| **Experiment Tracking** | **MLflow** | Tracks hyperparameters, execution metrics (RMSE, MAE, R²), diagnostic plots, and pipeline code. |
| **Model Registry** | **MLflow** | Manages trained model artifacts, versions, and production deployment aliases (`@champion`). |

### Exact Commands to Reproduce & Restore Dataset

1. **Verify DVC Status:**
   ```bash
   dvc status
   ```
2. **Push Data to Local DVC Remote:**
   ```bash
   dvc push
   ```
3. **Restore Dataset from DVC Remote:**
   ```bash
   dvc checkout
   # OR
   dvc pull
   ```

---

## 🧪 MLOps with MLflow

MLflow is integrated into the training pipeline to provide end-to-end experiment tracking, artifact logging, and model lifecycle management.

### Key MLflow Concepts Implemented
* **Why MLflow is Used:** In production machine learning, reproducibility, auditing, and experiment comparison are critical. MLflow eliminates untracked manual modeling by automatically recording hyperparameters, performance metrics, code versions, and trained artifacts for every execution.
* **Experiment Tracking:** All candidate model runs are logged under the experiment `Energy Consumption Prediction`.
* **Parameter Tracking:** For each candidate model (Linear Regression, Random Forest, XGBoost), hyperparameter configurations (e.g., `n_estimators`, `max_depth`, `learning_rate`, `random_state`) and dataset parameters (`dataset_name`, `target_column`, `train_rows`, `test_rows`, `feature_count`) are logged.
* **Metric Tracking:** Evaluation metrics computed on the chronological test split—**MAE**, **MSE**, **RMSE**, and **R²**—are logged for every run.
* **Artifact Tracking:** Each run logs the trained Scikit-learn/XGBoost `Pipeline` model artifact along with generated diagnostic figures (`actual_vs_predicted.png`, `residual_plot.png`, `prediction_error_distribution.png`, `actual_vs_predicted_time.png`, and `feature_importance.png`).
* **Model Registry & Versioning:** The best performing model (determined automatically by lowest test RMSE) is registered in the centralized MLflow Model Registry under the name `EnergyConsumptionModel`. Each new registration increments the model version.
* **Champion Alias Concept:** Rather than using deprecated stage names (e.g. "Production"), the top-performing registered model version is assigned the modern MLflow alias `champion`, allowing downstream inference services to retrieve the active production model via `models:/EnergyConsumptionModel@champion`.

### How to Launch & Explore MLflow UI

1. **Run Training Pipeline:**
   ```bash
   python -m src.train
   ```
2. **Start MLflow UI Server:**
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```
   *(Or simply run `mlflow ui` if MLflow default tracking is configured)*

3. **Access MLflow Interface:**
   Open your browser and navigate to:
   [http://127.0.0.1:5000](http://127.0.0.1:5000)

4. **Explore & Compare:**
   * Navigate to the experiment **`Energy Consumption Prediction`**.
   * Compare runs across models using scatter plots and metric comparison tables.
   * View registered models under **Models -> EnergyConsumptionModel** to inspect versions, run history, and the active `champion` model alias.

---

## ⚡ FastAPI Model Serving

A production-grade RESTful API built with **FastAPI** serves energy consumption predictions powered by the active MLflow registered model (`models:/EnergyConsumptionModel@champion`).

### Architecture Integration & Model Loading
```
+------------------+     +--------------------+     +------------------------+     +-------------------+
|  DVC Raw Data    | --> | MLflow Experiments | --> | MLflow Model Registry  | --> |  FastAPI Server   |
| (Versioned Data) |     |  (Metrics & Logs)  |     | (EnergyConsumption...  |     | (GET /, /health,  |
+------------------+     +--------------------+     |   alias: champion)     |     |  POST /predict)   |
                                                    +------------------------+     +-------------------+
```
* **Why FastAPI is Used:** FastAPI provides high-performance asynchronous request handling, automatic OpenAPI/Swagger schema documentation, and robust type validation using Pydantic.
* **Single-Load Startup (Lifespan):** The MLflow `champion` model is loaded once into memory during application startup using FastAPI's `lifespan` context manager, ensuring ultra-fast `POST /predict` inference without per-request model loading overhead.
* **Feature Schema Validation:** `api/schemas.py` enforces validation across all 41 model input features (indoor/outdoor microclimate data, temporal calendar encodings, target lag statistics, and rolling window averages).

### Key Endpoints

| Endpoint | Method | Description | Response Example |
| :--- | :--- | :--- | :--- |
| **`/`** | `GET` | API root overview & documentation links | `{"project": "Energy Consumption...", "status": "online"}` |
| **`/health`** | `GET` | Health status and MLflow model readiness check | `{"status": "healthy", "model_loaded": true}` |
| **`/predict`** | `POST` | Accepts feature JSON payload and returns forecast | `{"predicted_consumption": 64.01, "model_name": "...", "model_alias": "champion"}` |
| **`/docs`** | `GET` | Interactive Swagger UI documentation | HTML Interface |

### How to Start the API & Access Documentation

1. **Start Server with Uvicorn:**
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```
2. **Access Interactive Swagger UI:**
   Open browser at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

3. **Example Prediction Request (`POST /predict`):**
   ```json
   {
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
   ```

4. **Example Response:**
   ```json
   {
     "predicted_consumption": 64.01,
     "model_name": "EnergyConsumptionModel",
     "model_alias": "champion"
   }
   ```

---

## 🐳 Docker Containerization

The FastAPI model serving application and MLflow registered model resolution system are packaged into a production-oriented, reproducible Docker container.

### Architecture Overview
```
+---------------+     +------------------+     +-------------------+     +----------------------+     +------------------+
| DVC Dataset   | --> | MLflow Registry  | --> | FastAPI App       | --> | Docker Container     | --> | Prediction API   |
| (Raw Data v1) |     | (alias: champion)|     | (api/main.py)     |     | (Port 8000:8000)     |     | (POST /predict)  |
+---------------+     +------------------+     +-------------------+     +----------------------+     +------------------+
```
* **Why Docker is Used:** Docker eliminates "works on my machine" issues by isolating the application, dependencies, MLflow model registry artifacts, and Python environment into a lightweight, portable container.
* **Security & Non-Root User:** The container executes under a non-root security user (`appuser`).
* **Model Artifact Resolution inside Container:** The Docker build packages `mlflow.db` and the `mlruns/` artifact repository so `models:/EnergyConsumptionModel@champion` resolves seamlessly without requiring external cloud storage or remote servers.
* **Health Monitoring:** Implements an automated Docker `HEALTHCHECK` querying `http://localhost:8000/health` every 30 seconds.

### Exact Commands to Build and Run Container

1. **Build Docker Image:**
   ```bash
   docker build -t energy-consumption-mlops .
   ```

2. **Run Docker Container:**
   ```bash
   docker run -d -p 8000:8000 --name energy_api energy-consumption-mlops
   ```

3. **Orchestrate via Docker Compose:**
   ```bash
   docker-compose up -d
   ```

4. **Verify Container Logs:**
   ```bash
   docker logs energy_api
   ```

5. **Stop & Remove Container:**
   ```bash
   docker stop energy_api && docker rm energy_api
   ```

---

## 🔄 GitHub Actions CI/CD Pipeline

The project includes an automated Continuous Integration (CI) pipeline configured via GitHub Actions in `.github/workflows/ci.yml`.

### Workflow Triggers
* **Push** events to the `master` branch.
* **Pull Request** events targeting the `master` branch.

### Pipeline Steps & Environment
1. **Checkout Code:** Retrieves repository files (`actions/checkout@v4`).
2. **Set Up Python:** Configures Python 3.10 environment (`actions/setup-python@v5`).
3. **Install Dependencies:** Upgrades `pip` and installs dependencies from `requirements.txt`.
4. **Run Tests:** Executes the full pytest unit & integration test suite (`pytest`).
5. **Build Docker Image:** Builds the `energy-consumption-mlops:latest` container image to verify build integrity (`docker build -t energy-consumption-mlops:latest .`).




