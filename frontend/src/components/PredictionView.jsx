import React, { useState } from 'react';
import { Zap, Sliders, AlertTriangle, CheckCircle2, RefreshCw } from 'lucide-react';
import { DEFAULT_FEATURE_VALUES, FEATURE_GROUPS } from '../constants/featureSchema';
import { predictEnergy } from '../services/api';

export default function PredictionView({ health }) {
  const [formData, setFormData] = useState({ ...DEFAULT_FEATURE_VALUES });
  const [prediction, setPrediction] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [error, setError] = useState(null);

  const handleInputChange = (name, value) => {
    setFormData((prev) => ({
      ...prev,
      [name]: value === '' ? '' : Number(value),
    }));
  };

  const handleResetPreset = () => {
    setFormData({ ...DEFAULT_FEATURE_VALUES });
    setError(null);
  };

  const handlePredictSubmit = async (e) => {
    e.preventDefault();
    setIsPredicting(true);
    setError(null);

    try {
      const payload = {};
      Object.keys(formData).forEach((key) => {
        payload[key] = Number(formData[key]);
      });

      const res = await predictEnergy(payload);
      setPrediction(res);
    } catch (err) {
      setError(err.message || 'Prediction failed. Check backend connection.');
    } finally {
      setIsPredicting(false);
    }
  };

  const isHealthy = health && health.status === 'healthy';

  return (
    <div className="prediction-view">
      <div className="form-section">
        <div className="section-header" style={{ justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <Zap size={24} color="#60a5fa" />
            <div>
              <h2 className="section-title">Appliance Energy Consumption Prediction</h2>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                Full feature input form generated strictly from FastAPI <code style={{ color: '#93c5fd' }}>/predict</code> schema (41 Features)
              </p>
            </div>
          </div>

          <button type="button" className="btn btn-secondary" onClick={handleResetPreset}>
            <Sliders size={16} />
            <span>Reset to Benchmark Preset</span>
          </button>
        </div>

        <form onSubmit={handlePredictSubmit}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            {FEATURE_GROUPS.map((group) => (
              <div key={group.id} style={{ background: 'rgba(0,0,0,0.25)', padding: '1.5rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
                <h3 style={{ fontSize: '1rem', color: '#93c5fd', marginBottom: '0.25rem', fontWeight: 700 }}>
                  {group.name}
                </h3>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-dim)', marginBottom: '1.25rem' }}>
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
              style={{ width: '100%', maxWidth: '450px', padding: '1.1rem', fontSize: '1rem' }}
            >
              {isPredicting ? (
                <>
                  <RefreshCw size={20} className="spin" />
                  <span>Computing Prediction...</span>
                </>
              ) : (
                <>
                  <Zap size={20} />
                  <span>[ Predict Energy Consumption ]</span>
                </>
              )}
            </button>
          </div>
        </form>

        {error && (
          <div className="error-banner" style={{ marginTop: '1.5rem' }}>
            <AlertTriangle size={20} />
            <span>{error}</span>
          </div>
        )}

        {prediction && (
          <div className="prediction-result-card" style={{ marginTop: '2rem' }}>
            <div className="prediction-title">PREDICTED ENERGY CONSUMPTION</div>
            <div className="prediction-number">
              {prediction.predicted_consumption} <span className="prediction-unit">Wh</span>
            </div>
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', background: 'rgba(16, 185, 129, 0.15)', color: '#34d399', padding: '0.4rem 1rem', borderRadius: 'var(--radius-full)', fontSize: '0.85rem', fontWeight: 600, border: '1px solid rgba(16,185,129,0.3)' }}>
              <CheckCircle2 size={16} />
              <span>Served by {prediction.model_name} (@{prediction.model_alias})</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
