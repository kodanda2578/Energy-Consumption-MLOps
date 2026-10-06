import React, { useState } from 'react';
import { 
  ShieldCheck, 
  Cpu, 
  Tag, 
  Activity, 
  Zap, 
  RefreshCw, 
  AlertTriangle, 
  BarChart2, 
  CheckCircle2,
  Sliders
} from 'lucide-react';
import { DEFAULT_FEATURE_VALUES, FEATURE_GROUPS, MODEL_COMPARISON_METRICS } from '../constants/featureSchema';
import { predictEnergy } from '../services/api';

export default function DashboardView({ health, monitoring, error }) {
  const [formData, setFormData] = useState({ ...DEFAULT_FEATURE_VALUES });
  const [prediction, setPrediction] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictError, setPredictError] = useState(null);

  const handleInputChange = (name, value) => {
    setFormData((prev) => ({
      ...prev,
      [name]: value === '' ? '' : Number(value),
    }));
  };

  const handlePresetData = () => {
    setFormData({ ...DEFAULT_FEATURE_VALUES });
    setPredictError(null);
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setIsPredicting(true);
    setPredictError(null);

    try {
      // Ensure numeric types across payload
      const formattedPayload = {};
      Object.keys(formData).forEach((key) => {
        formattedPayload[key] = Number(formData[key]);
      });

      const res = await predictEnergy(formattedPayload);
      setPrediction(res);
    } catch (err) {
      setPredictError(err.message || 'Prediction failed. Check FastAPI backend logs.');
    } finally {
      setIsPredicting(false);
    }
  };

  const isHealthy = health && health.status === 'healthy';
  const hasDrift = monitoring && monitoring.overall_drift_detected;

  return (
    <div className="dashboard-view">
      {/* System Error Notification if Backend Offline */}
      {error && (
        <div className="error-banner">
          <AlertTriangle size={20} />
          <span>Backend Connection Error: {error}. Please ensure FastAPI backend is running on http://localhost:8000.</span>
        </div>
      )}

      {/* 2. Top Summary Cards */}
      <div className="card-grid">
        {/* Card 1: Model Status */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Model Status</span>
            <div className="card-icon-wrapper" style={{ color: isHealthy ? '#34d399' : '#f87171' }}>
              <ShieldCheck size={20} />
            </div>
          </div>
          <div className="card-value" style={{ color: isHealthy ? '#34d399' : '#f87171' }}>
            {isHealthy ? 'Healthy' : 'Unhealthy'}
          </div>
          <div className="card-subtext">
            {health?.model_loaded ? 'Champion model ready in memory' : 'Model not loaded'}
          </div>
        </div>

        {/* Card 2: Champion Model */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Champion Model</span>
            <div className="card-icon-wrapper" style={{ color: '#60a5fa' }}>
              <Cpu size={20} />
            </div>
          </div>
          <div className="card-value" style={{ fontSize: '1.25rem', color: '#93c5fd' }}>
            {health?.model_name || 'EnergyConsumptionModel'}
          </div>
          <div className="card-subtext">MLflow Registered Registry</div>
        </div>

        {/* Card 3: Model Alias */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Model Alias</span>
            <div className="card-icon-wrapper" style={{ color: '#c084fc' }}>
              <Tag size={20} />
            </div>
          </div>
          <div className="card-value" style={{ color: '#c084fc' }}>
            {health?.model_alias || 'champion'}
          </div>
          <div className="card-subtext">Production Alias Tag</div>
        </div>

        {/* Card 4: Data Drift */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Data Drift</span>
            <div className="card-icon-wrapper" style={{ color: hasDrift ? '#f87171' : '#34d399' }}>
              <Activity size={20} />
            </div>
          </div>
          <div className="card-value" style={{ color: hasDrift ? '#f87171' : '#34d399', fontSize: '1.4rem' }}>
            {hasDrift ? 'Drift Detected' : 'No Drift'}
          </div>
          <div className="card-subtext">
            {monitoring ? `${monitoring.drifted_features || 0} of ${monitoring.total_features || 41} features drifted` : 'KS Test evaluation'}
          </div>
        </div>
      </div>

      {/* 3. Prediction Section */}
      <div className="form-section">
        <div className="section-header" style={{ justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <Zap size={22} color="#60a5fa" />
            <div>
              <h2 className="section-title">Appliance Energy Forecast</h2>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                Input feature values derived from dataset telemetry (41 Features)
              </div>
            </div>
          </div>

          <button type="button" className="btn btn-secondary" onClick={handlePresetData}>
            <Sliders size={15} />
            <span>Load Preset Sample Data</span>
          </button>
        </div>

        <form onSubmit={handlePredict}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            {FEATURE_GROUPS.map((group) => (
              <div key={group.id} style={{ background: 'rgba(0,0,0,0.2)', padding: '1.25rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
                <h4 style={{ fontSize: '0.9rem', color: '#93c5fd', marginBottom: '0.25rem', fontWeight: 700 }}>
                  {group.name}
                </h4>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginBottom: '1rem' }}>
                  {group.description}
                </p>

                <div className="form-grid">
                  {group.fields.map((field) => (
                    <div key={field.name} className="form-group">
                      <label className="form-label">{field.label}</label>
                      <input
                        type={field.type}
                        step={field.step}
                        min={field.min}
                        max={field.max}
                        className="form-input"
                        value={formData[field.name] !== undefined ? formData[field.name] : ''}
                        onChange={(e) => handleInputChange(field.name, e.target.value)}
                        required
                      />
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          <div style={{ marginTop: '2rem', display: 'flex', justifyContent: 'center' }}>
            <button 
              type="submit" 
              className="btn btn-primary" 
              disabled={isPredicting || !isHealthy}
              style={{ width: '100%', maxWidth: '400px', padding: '1rem' }}
            >
              {isPredicting ? (
                <>
                  <RefreshCw size={18} className="spin" />
                  <span>Predicting...</span>
                </>
              ) : (
                <>
                  <Zap size={18} />
                  <span>[ Predict Energy Consumption ]</span>
                </>
              )}
            </button>
          </div>
        </form>

        {/* Prediction Error Alert */}
        {predictError && (
          <div className="error-banner" style={{ marginTop: '1.5rem' }}>
            <AlertTriangle size={20} />
            <span>{predictError}</span>
          </div>
        )}

        {/* Prediction Result Display Box */}
        {prediction && (
          <div className="prediction-result-card">
            <div className="prediction-title">FORECASTED ENERGY CONSUMPTION</div>
            <div className="prediction-number">
              {prediction.predicted_consumption} <span className="prediction-unit">Wh</span>
            </div>
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', background: 'rgba(16, 185, 129, 0.15)', color: '#34d399', padding: '0.3rem 0.8rem', borderRadius: 'var(--radius-full)', fontSize: '0.8rem', fontWeight: 600, border: '1px solid rgba(16,185,129,0.3)' }}>
              <CheckCircle2 size={15} />
              <span>Served by {prediction.model_name} (@{prediction.model_alias})</span>
            </div>
          </div>
        )}
      </div>

      {/* 4. Monitoring & 5. System Health Section */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.5rem' }}>
        {/* Data Drift Summary Card */}
        <div className="form-section" style={{ marginBottom: 0 }}>
          <div className="section-header">
            <Activity size={20} color="#34d399" />
            <h3 className="section-title">Data Drift Summary (GET /monitoring)</h3>
          </div>

          <table className="custom-table">
            <tbody>
              <tr>
                <td>Total Monitored Features</td>
                <td style={{ fontWeight: 700, fontFamily: 'monospace' }}>{monitoring?.total_features ?? 41}</td>
              </tr>
              <tr>
                <td>Drifted Features Count</td>
                <td style={{ fontWeight: 700, color: (monitoring?.drifted_features ?? 0) > 0 ? '#f87171' : '#34d399' }}>
                  {monitoring?.drifted_features ?? 0}
                </td>
              </tr>
              <tr>
                <td>Drift Percentage</td>
                <td style={{ fontWeight: 700 }}>{monitoring?.drift_percentage ?? 0.0}%</td>
              </tr>
              <tr>
                <td>Significance Threshold (Alpha)</td>
                <td style={{ fontFamily: 'monospace' }}>α = {monitoring?.alpha ?? 0.05}</td>
              </tr>
              <tr>
                <td>Overall Status</td>
                <td>
                  <span className={`status-pill ${hasDrift ? 'unhealthy' : 'healthy'}`}>
                    {hasDrift ? '⚠ Drift Detected' : '✔ No Drift Detected'}
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* System Health Card */}
        <div className="form-section" style={{ marginBottom: 0 }}>
          <div className="section-header">
            <ShieldCheck size={20} color="#60a5fa" />
            <h3 className="section-title">System Health (GET /health)</h3>
          </div>

          <table className="custom-table">
            <tbody>
              <tr>
                <td>API Status</td>
                <td>
                  <span className={`status-pill ${isHealthy ? 'healthy' : 'unhealthy'}`}>
                    {isHealthy ? 'Healthy' : 'Unhealthy'}
                  </span>
                </td>
              </tr>
              <tr>
                <td>Model Loaded</td>
                <td style={{ fontWeight: 700, color: health?.model_loaded ? '#34d399' : '#f87171' }}>
                  {health?.model_loaded ? 'True (In Memory)' : 'False'}
                </td>
              </tr>
              <tr>
                <td>Model Name</td>
                <td style={{ fontFamily: 'monospace' }}>{health?.model_name || 'EnergyConsumptionModel'}</td>
              </tr>
              <tr>
                <td>Model Alias</td>
                <td style={{ fontFamily: 'monospace' }}>{health?.model_alias || 'champion'}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
