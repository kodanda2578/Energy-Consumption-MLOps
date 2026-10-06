/**
 * Exact schema metadata and benchmark sample values for all 41 model features.
 * Matches Pydantic EnergyPredictionRequest in api/schemas.py.
 */

export const DEFAULT_FEATURE_VALUES = {
  lights: 0,
  T1: 19.89, RH_1: 47.59,
  T2: 19.20, RH_2: 44.79,
  T3: 19.79, RH_3: 44.73,
  T4: 19.00, RH_4: 45.56,
  T5: 17.1667, RH_5: 55.20,
  T6: 7.0267, RH_6: 84.2567,
  T7: 17.20, RH_7: 41.6267,
  T8: 18.20, RH_8: 48.90,
  T9: 17.0333, RH_9: 45.53,
  T_out: 6.60, Press_mm_hg: 733.50, RH_out: 92.00,
  Windspeed: 7.00, Visibility: 63.00, Tdewpoint: 5.30,
  hour: 18, day_of_week: 0, month: 1, day: 11, is_weekend: 0,
  sin_hour: -1.00, cos_hour: 0.00,
  sin_day_of_week: 0.00, cos_day_of_week: 1.00,
  appliances_lag_1: 60.0, appliances_lag_3: 50.0,
  appliances_lag_6: 50.0, appliances_lag_12: 60.0,
  rolling_mean_3: 53.3333, rolling_mean_6: 55.0, rolling_mean_12: 58.3333
};

export const FEATURE_GROUPS = [
  {
    id: 'indoor_climate',
    name: 'Indoor Microclimate',
    icon: 'Home',
    description: 'Temperature (°C) and Relative Humidity (%) across 9 monitored rooms',
    fields: [
      { name: 'T1', label: 'T1 - Kitchen Temp', unit: '°C', type: 'number', step: '0.1' },
      { name: 'RH_1', label: 'RH_1 - Kitchen Humidity', unit: '%', type: 'number', step: '0.1' },
      { name: 'T2', label: 'T2 - Living Room Temp', unit: '°C', type: 'number', step: '0.1' },
      { name: 'RH_2', label: 'RH_2 - Living Room Humidity', unit: '%', type: 'number', step: '0.1' },
      { name: 'T3', label: 'T3 - Laundry Temp', unit: '°C', type: 'number', step: '0.1' },
      { name: 'RH_3', label: 'RH_3 - Laundry Humidity', unit: '%', type: 'number', step: '0.1' },
      { name: 'T4', label: 'T4 - Office Temp', unit: '°C', type: 'number', step: '0.1' },
      { name: 'RH_4', label: 'RH_4 - Office Humidity', unit: '%', type: 'number', step: '0.1' },
      { name: 'T5', label: 'T5 - Bathroom Temp', unit: '°C', type: 'number', step: '0.1' },
      { name: 'RH_5', label: 'RH_5 - Bathroom Humidity', unit: '%', type: 'number', step: '0.1' },
      { name: 'T6', label: 'T6 - Outside North Temp', unit: '°C', type: 'number', step: '0.1' },
      { name: 'RH_6', label: 'RH_6 - Outside North Humidity', unit: '%', type: 'number', step: '0.1' },
      { name: 'T7', label: 'T7 - Ironing Room Temp', unit: '°C', type: 'number', step: '0.1' },
      { name: 'RH_7', label: 'RH_7 - Ironing Room Humidity', unit: '%', type: 'number', step: '0.1' },
      { name: 'T8', label: 'T8 - Teenager Room Temp', unit: '°C', type: 'number', step: '0.1' },
      { name: 'RH_8', label: 'RH_8 - Teenager Room Humidity', unit: '%', type: 'number', step: '0.1' },
      { name: 'T9', label: 'T9 - Parents Room Temp', unit: '°C', type: 'number', step: '0.1' },
      { name: 'RH_9', label: 'RH_9 - Parents Room Humidity', unit: '%', type: 'number', step: '0.1' },
    ]
  },
  {
    id: 'outdoor_weather',
    name: 'Outdoor Weather',
    icon: 'CloudSun',
    description: 'Meteorological telemetry from airport station',
    fields: [
      { name: 'T_out', label: 'Outdoor Temperature', unit: '°C', type: 'number', step: '0.1' },
      { name: 'Press_mm_hg', label: 'Barometric Pressure', unit: 'mmHg', type: 'number', step: '0.1' },
      { name: 'RH_out', label: 'Outdoor Humidity', unit: '%', type: 'number', step: '0.1' },
      { name: 'Windspeed', label: 'Wind Speed', unit: 'm/s', type: 'number', step: '0.1' },
      { name: 'Visibility', label: 'Visibility', unit: 'km', type: 'number', step: '0.1' },
      { name: 'Tdewpoint', label: 'Dew Point Temp', unit: '°C', type: 'number', step: '0.1' },
    ]
  },
  {
    id: 'temporal',
    name: 'Temporal & Calendar',
    icon: 'Calendar',
    description: 'Date, time, weekend flag, and cyclical sine/cosine encodings',
    fields: [
      { name: 'hour', label: 'Hour of Day (0-23)', unit: 'h', type: 'number', step: '1', min: 0, max: 23 },
      { name: 'day_of_week', label: 'Day of Week (0=Mon)', unit: 'd', type: 'number', step: '1', min: 0, max: 6 },
      { name: 'month', label: 'Month (1-12)', unit: 'm', type: 'number', step: '1', min: 1, max: 12 },
      { name: 'day', label: 'Day of Month (1-31)', unit: 'd', type: 'number', step: '1', min: 1, max: 31 },
      { name: 'is_weekend', label: 'Is Weekend (0 or 1)', unit: 'flag', type: 'number', step: '1', min: 0, max: 1 },
      { name: 'sin_hour', label: 'Sine Hour Encoding', unit: 'val', type: 'number', step: '0.01', min: -1, max: 1 },
      { name: 'cos_hour', label: 'Cosine Hour Encoding', unit: 'val', type: 'number', step: '0.01', min: -1, max: 1 },
      { name: 'sin_day_of_week', label: 'Sine Day Encoding', unit: 'val', type: 'number', step: '0.01', min: -1, max: 1 },
      { name: 'cos_day_of_week', label: 'Cosine Day Encoding', unit: 'val', type: 'number', step: '0.01', min: -1, max: 1 },
    ]
  },
  {
    id: 'lags_rolling',
    name: 'Historical Lags & Rolling Window',
    icon: 'History',
    description: 'Historical target appliance consumption (t-1, t-3, t-6, t-12) & rolling averages',
    fields: [
      { name: 'appliances_lag_1', label: 'Appliances Lag 1 (t-10min)', unit: 'Wh', type: 'number', step: '1' },
      { name: 'appliances_lag_3', label: 'Appliances Lag 3 (t-30min)', unit: 'Wh', type: 'number', step: '1' },
      { name: 'appliances_lag_6', label: 'Appliances Lag 6 (t-1h)', unit: 'Wh', type: 'number', step: '1' },
      { name: 'appliances_lag_12', label: 'Appliances Lag 12 (t-2h)', unit: 'Wh', type: 'number', step: '1' },
      { name: 'rolling_mean_3', label: '3-Period Rolling Mean', unit: 'Wh', type: 'number', step: '0.1' },
      { name: 'rolling_mean_6', label: '6-Period Rolling Mean', unit: 'Wh', type: 'number', step: '0.1' },
      { name: 'rolling_mean_12', label: '12-Period Rolling Mean', unit: 'Wh', type: 'number', step: '0.1' },
    ]
  },
  {
    id: 'lighting',
    name: 'Lighting & Loads',
    icon: 'Zap',
    description: 'Sub-metered lighting consumption',
    fields: [
      { name: 'lights', label: 'Lights Consumption', unit: 'Wh', type: 'number', step: '1', min: 0 },
    ]
  }
];

export const MODEL_COMPARISON_METRICS = [
  { model: 'Linear Regression', mae: 27.63, rmse: 59.77, r2: 0.5650, status: 'Best Model ⭐' },
  { model: 'XGBoost Regressor', mae: 50.36, rmse: 78.26, r2: 0.2541, status: 'Candidate' },
  { model: 'Random Forest Regressor', mae: 69.23, rmse: 103.39, r2: -0.3018, status: 'Candidate' }
];
