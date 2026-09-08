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
- [ ] **Phase 6: API Development & Streamlit Frontend**
  - RESTful API endpoints via FastAPI
  - Interactive Web App UI built with Streamlit
- [ ] **Phase 7: Containerization, CI/CD & Monitoring**
  - Dockerization of API & UI services
  - GitHub Actions CI workflow
  - Prometheus/Grafana lightweight performance monitoring

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


