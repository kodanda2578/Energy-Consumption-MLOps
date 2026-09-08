"""
Pydantic Request and Response Schemas for Energy Consumption Prediction API.
"""

from pydantic import BaseModel, Field
from typing import Optional


class EnergyPredictionRequest(BaseModel):
    """
    Input feature schema for Energy Consumption Prediction.
    Contains all 41 features required by the trained model pipeline.
    """
    lights: float = Field(..., ge=0, description="Lighting energy consumption in Wh")
    T1: float = Field(..., description="Kitchen temperature in °C")
    RH_1: float = Field(..., description="Kitchen relative humidity in %")
    T2: float = Field(..., description="Living room temperature in °C")
    RH_2: float = Field(..., description="Living room relative humidity in %")
    T3: float = Field(..., description="Laundry room temperature in °C")
    RH_3: float = Field(..., description="Laundry room relative humidity in %")
    T4: float = Field(..., description="Office room temperature in °C")
    RH_4: float = Field(..., description="Office room relative humidity in %")
    T5: float = Field(..., description="Bathroom temperature in °C")
    RH_5: float = Field(..., description="Bathroom relative humidity in %")
    T6: float = Field(..., description="Outside north temperature in °C")
    RH_6: float = Field(..., description="Outside north relative humidity in %")
    T7: float = Field(..., description="Ironing room temperature in °C")
    RH_7: float = Field(..., description="Ironing room relative humidity in %")
    T8: float = Field(..., description="Teenager room temperature in °C")
    RH_8: float = Field(..., description="Teenager room relative humidity in %")
    T9: float = Field(..., description="Parents room temperature in °C")
    RH_9: float = Field(..., description="Parents room relative humidity in %")
    T_out: float = Field(..., description="Outdoor temperature in °C")
    Press_mm_hg: float = Field(..., description="Barometric pressure in mm Hg")
    RH_out: float = Field(..., description="Outdoor relative humidity in %")
    Windspeed: float = Field(..., description="Wind speed in m/s")
    Visibility: float = Field(..., description="Visibility in km")
    Tdewpoint: float = Field(..., description="Dew point temperature in °C")
    hour: int = Field(..., ge=0, le=23, description="Hour of the day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    month: int = Field(..., ge=1, le=12, description="Month of year (1-12)")
    day: int = Field(..., ge=1, le=31, description="Day of month (1-31)")
    is_weekend: int = Field(..., ge=0, le=1, description="Is weekend flag (0 or 1)")
    sin_hour: float = Field(..., ge=-1.0, le=1.0, description="Cyclical sine encoding of hour")
    cos_hour: float = Field(..., ge=-1.0, le=1.0, description="Cyclical cosine encoding of hour")
    sin_day_of_week: float = Field(..., ge=-1.0, le=1.0, description="Cyclical sine encoding of day of week")
    cos_day_of_week: float = Field(..., ge=-1.0, le=1.0, description="Cyclical cosine encoding of day of week")
    appliances_lag_1: float = Field(..., ge=0, description="Appliance consumption 1 interval ago (t-1)")
    appliances_lag_3: float = Field(..., ge=0, description="Appliance consumption 3 intervals ago (t-3)")
    appliances_lag_6: float = Field(..., ge=0, description="Appliance consumption 6 intervals ago (t-6)")
    appliances_lag_12: float = Field(..., ge=0, description="Appliance consumption 12 intervals ago (t-12)")
    rolling_mean_3: float = Field(..., ge=0, description="3-period rolling average of target")
    rolling_mean_6: float = Field(..., ge=0, description="6-period rolling average of target")
    rolling_mean_12: float = Field(..., ge=0, description="12-period rolling average of target")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class EnergyPredictionResponse(BaseModel):
    """
    Response schema for model prediction endpoint.
    """
    predicted_consumption: float = Field(..., description="Predicted appliance energy consumption in Wh")
    model_name: str = Field(..., description="Name of the registered MLflow model")
    model_alias: str = Field(..., description="MLflow model version alias (e.g. champion)")


class HealthResponse(BaseModel):
    """
    Response schema for service health endpoint.
    """
    status: str = Field(..., description="API operational status (healthy/unhealthy)")
    model_loaded: bool = Field(..., description="Whether the champion MLflow model is loaded and ready")
    model_name: str = Field(..., description="Name of the loaded model")
    model_alias: str = Field(..., description="Alias of the loaded model")
