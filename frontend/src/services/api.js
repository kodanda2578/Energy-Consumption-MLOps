const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Fetch service health status from GET /health
 */
export async function getHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    if (!response.ok) {
      throw new Error(`Health check failed with status ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('[API Service] Health check error:', error);
    throw error;
  }
}

/**
 * Fetch latest data drift monitoring summary from GET /monitoring
 */
export async function getMonitoring() {
  try {
    const response = await fetch(`${API_BASE_URL}/monitoring`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    if (!response.ok) {
      throw new Error(`Monitoring query failed with status ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('[API Service] Monitoring query error:', error);
    throw error;
  }
}

/**
 * Post feature payload to POST /predict for energy consumption forecast
 * @param {Object} payload All 41 feature inputs matching EnergyPredictionRequest schema
 */
export async function predictEnergy(payload) {
  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorJson = await response.json().catch(() => ({}));
      throw new Error(errorJson.detail || `Prediction request failed with status ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('[API Service] Prediction error:', error);
    throw error;
  }
}
