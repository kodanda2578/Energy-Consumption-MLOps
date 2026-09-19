"""
Evaluation and Visualization Module for Energy Consumption Pipeline.

Calculates regression performance metrics (MAE, MSE, RMSE, R²) and generates
diagnostic diagnostic plots saved in reports/figures/.
"""

import sys
import os

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data_loader import load_config


def evaluate_predictions(y_true, y_pred) -> dict:
    """
    Compute regression evaluation metrics.

    Returns:
        dict: Containing MAE, MSE, RMSE, and R2 metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
    }


def generate_evaluation_plots(y_true, y_pred, model_name: str, config_path: str = "configs/config.yaml"):
    """
    Generate and save diagnostic evaluation plots:
    1. Actual vs Predicted Scatter Plot
    2. Residuals vs Predicted Plot
    3. Prediction Error Distribution
    4. Time-series Actual vs Predicted Line Plot (sample)
    """
    config = load_config(config_path)
    figures_dir = config.get("reports", {}).get("figures_dir", "reports/figures")
    os.makedirs(figures_dir, exist_ok=True)

    residuals = y_true - y_pred

    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, color="teal")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
    plt.xlabel("Actual Energy Consumption (Wh)")
    plt.ylabel("Predicted Energy Consumption (Wh)")
    plt.title(f"Actual vs Predicted - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "actual_vs_predicted.png"))
    plt.close()

    # Plot 2: Residual Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.3, color="coral")
    plt.axhline(0, color="black", linestyle="--", lw=1.5)
    plt.xlabel("Predicted Energy Consumption (Wh)")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title(f"Residual Plot - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "residual_plot.png"))
    plt.close()

    # Plot 3: Prediction Error Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color="purple", bins=30)
    plt.xlabel("Prediction Error (Wh)")
    plt.title(f"Prediction Error Distribution - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "prediction_error_distribution.png"))
    plt.close()

    # Plot 4: Time Series Overlay (Sample 200 time steps)
    plt.figure(figsize=(12, 5))
    sample_len = min(200, len(y_true))
    plt.plot(np.array(y_true)[:sample_len], label="Actual Energy (Wh)", color="blue", alpha=0.8)
    plt.plot(np.array(y_pred)[:sample_len], label="Predicted Energy (Wh)", color="orange", linestyle="--", alpha=0.8)
    plt.xlabel("Time Step (Sampled 10-min Intervals)")
    plt.ylabel("Energy Consumption (Wh)")
    plt.title(f"Actual vs Predicted Over Time (Sample) - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "actual_vs_predicted_time.png"))
    plt.close()

    print(f"[SUCCESS] Saved diagnostic plots to: {figures_dir}")


def generate_feature_importance_plot(model, feature_names: list, model_name: str, config_path: str = "configs/config.yaml") -> str:
    """
    Generate and save feature importance plot if model supports feature_importances_.

    Returns:
        str: Path to saved feature importance plot image or None.
    """
    if not hasattr(model, "feature_importances_"):
        return None

    config = load_config(config_path)
    figures_dir = config.get("reports", {}).get("figures_dir", "reports/figures")
    os.makedirs(figures_dir, exist_ok=True)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # Top 15 features

    plt.figure(figsize=(10, 6))
    plt.title(f"Top 15 Feature Importances - {model_name}")
    plt.bar(range(len(indices)), importances[indices], align="center", color="indigo")
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha="right")
    plt.ylabel("Relative Importance")
    plt.tight_layout()

    plot_path = os.path.join(figures_dir, "feature_importance.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path
