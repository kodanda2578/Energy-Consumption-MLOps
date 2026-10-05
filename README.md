# Energy Consumption Prediction with MLOps

## 📌 Project Objective
Build an end-to-end, production-style Machine Learning system that predicts energy consumption based on historical, time, weather, and energy usage features. The project follows software engineering and MLOps best practices—spanning data preprocessing, experiment tracking, model registry, API deployment, containerization, CI/CD automation, interactive user interface, and system monitoring.

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

## 📈 Phase 3 Baseline Model Performance Results

| Model | MAE (Wh) | RMSE (Wh) | R² Score | Status |
| :--- | ---: | ---: | ---: | :---: |
| **Linear Regression** ⭐ | **27.63** | **59.77** | **0.5650** | **Best Model Selected** |
| **XGBoost Regressor** | 50.36 | 78.26 | 0.2541 | Candidate |
| **Random Forest Regressor** | 69.23 | 103.39 | -0.3018 | Candidate |

*Linear Regression achieved the lowest RMSE (59.77) and highest R² (0.5650) on the chronological test set.*

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
* **Monitoring:** Prometheus & Grafana

---

## 🚀 How to Run the Pipeline

1. **Environment Setup**:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Data Acquisition & Loading**:
   ```bash
   python -m src.data_loader
   ```

3. **Run Data Preprocessing & Model Training**:
   ```bash
   python -m src.train
   ```

4. **Run Automated Test Suite**:
   ```bash
   pytest
   ```

---

## 🗺️ Development Roadmap

- [x] **Phase 1: Project Setup & Structure**
  - Modular directory design & VS Code configuration
  - Dependency definition (`requirements.txt`) & configuration template (`config.yaml`)
- [x] **Phase 2: Data Intake & Exploratory Data Analysis (EDA)**
  - Dataset evaluation & selection (UCI Appliance Energy Dataset)
  - Automated reproducible data loader module (`src/data_loader.py`)
  - Exploratory analysis notebook (`notebooks/01_exploratory_data_analysis.ipynb`)
- [x] **Phase 3: Data Preprocessing & Baseline Model Training**
  - Modular preprocessing pipeline (`src/preprocessing.py`)
  - Feature engineering (calendar, cyclical, lagged targets, rolling means)
  - Strict data leakage prevention & chronological train/test split
  - Model training (`Linear Regression`, `Random Forest`, `XGBoost`) & evaluation (`src/evaluate.py`)
  - Pipeline serialization (`models/energy_predictor.joblib`) & diagnostic plots (`reports/figures/`)
- [ ] **Phase 4: MLOps Integration (Experiment Tracking with MLflow & Versioning)**
  - Tracking hyperparameters and metrics with MLflow
  - Model serialization and registration
- [ ] **Phase 5: API Development & Streamlit Frontend**
  - RESTful API endpoints via FastAPI
  - Interactive Web App UI built with Streamlit
- [x] **Phase 6: Containerization & CI/CD**
  - Production-ready Dockerfile for FastAPI model serving
  - GitHub Actions CI/CD automation pipeline on Ubuntu runner
- [x] **Phase 9: Model Monitoring & Data Drift Detection**
  - Statistical Kolmogorov-Smirnov (KS) testing module (`src/monitoring.py`)
  - Configurable significance threshold ($\alpha = 0.05$)
  - Automated report generation (`drift_summary.json`, `drift_report.csv`, `drift_report.json`) & p-value visualization (`drift_pvalues.png`)
  - MLflow experiment tracking integration (`run_type = "data_drift_monitoring"`)
  - RESTful FastAPI monitoring endpoint (`GET /monitoring`)

---

## 🔍 Phase 9 — Monitoring & Data Drift

### Overview
In production Machine Learning systems, feature distributions change over time due to seasonal shifts, sensor calibration updates, or behavioral changes—a phenomenon known as **Data Drift**. When incoming production data diverges significantly from training data, model prediction performance degrades.

### Monitoring Architecture & Methodology
1. **Reference Data (Baseline):** The processed training feature distribution (`X_train` from `data/processed/energydata_processed.csv`).
2. **Current Data:** Incoming inference batches or new operational telemetry.
3. **Statistical Test (Kolmogorov-Smirnov Test):** For each numerical feature, a two-sample Kolmogorov-Smirnov (KS) test is performed (`scipy.stats.ks_2samp`) comparing reference vs. current distributions.
4. **Significance Threshold ($\alpha = 0.05$):** Configured via `configs/config.yaml`. A feature is flagged as drifted if its $p$-value satisfies:
   $$p\text{-value} < \alpha$$
5. **Drift Metrics & Summary:**
   * `total_features`: Number of evaluated numerical features
   * `drifted_features`: Count of features exhibiting statistically significant drift
   * `drift_percentage`: Percentage of total features drifted
   * `overall_drift_detected`: Boolean flag (`true` if `drifted_features > 0`)

### MLflow & FastAPI Integration
* **MLflow Tracking:** Logs monitoring runs under experiment `"Energy Consumption Prediction"` with tag `run_type = "data_drift_monitoring"`, metrics, and report artifacts.
* **FastAPI Endpoint:** `GET /monitoring` returns real-time data drift summary JSON:
  ```json
  {
    "status": "success",
    "monitoring": {
      "total_features": 41,
      "drifted_features": 0,
      "drift_percentage": 0.0,
      "overall_drift_detected": false,
      "alpha": 0.05,
      "timestamp": "2026-09-22T06:12:28.212025+00:00"
    }
  }
  ```
* **Monitoring Reports:** Generated on demand under `reports/monitoring/` (`drift_summary.json`, `drift_report.csv`, `drift_report.json`, and `drift_pvalues.png`).

