import React from 'react';
import { Cpu, Award, BarChart2, ShieldCheck, Database } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend, CartesianGrid } from 'recharts';
import { MODEL_COMPARISON_METRICS } from '../constants/featureSchema';

export default function ModelInfoView({ health }) {
  const bestModel = MODEL_COMPARISON_METRICS.find((m) => m.status.includes('Best'));

  return (
    <div className="model-info-view">
      <div className="form-section">
        <div className="section-header">
          <Cpu size={24} color="#60a5fa" />
          <div>
            <h2 className="section-title">Champion Model Information & Metrics</h2>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              MLflow registered champion pipeline details and phase evaluation benchmarks
            </p>
          </div>
        </div>

        {/* Champion Model Overview Header */}
        <div className="card-grid">
          <div className="card">
            <div className="card-header">
              <span className="card-title">Registry Model Name</span>
              <Database size={20} color="#60a5fa" />
            </div>
            <div className="card-value" style={{ fontSize: '1.2rem', color: '#93c5fd' }}>
              {health?.model_name || 'EnergyConsumptionModel'}
            </div>
            <div className="card-subtext">MLflow Tracking Artifact</div>
          </div>

          <div className="card">
            <div className="card-header">
              <span className="card-title">Active Alias Tag</span>
              <ShieldCheck size={20} color="#c084fc" />
            </div>
            <div className="card-value" style={{ color: '#c084fc' }}>
              {health?.model_alias || 'champion'}
            </div>
            <div className="card-subtext">Served via FastAPI /predict</div>
          </div>

          <div className="card">
            <div className="card-header">
              <span className="card-title">Best Architecture</span>
              <Award size={20} color="#f59e0b" />
            </div>
            <div className="card-value" style={{ fontSize: '1.25rem', color: '#f59e0b' }}>
              Linear Regression
            </div>
            <div className="card-subtext">Selected Best Performer</div>
          </div>

          <div className="card">
            <div className="card-header">
              <span className="card-title">Chronological R² Score</span>
              <BarChart2 size={20} color="#34d399" />
            </div>
            <div className="card-value" style={{ color: '#34d399' }}>
              {bestModel?.r2 ?? 0.5650}
            </div>
            <div className="card-subtext">Test Set R-Squared</div>
          </div>
        </div>

        {/* Detailed Metrics Table */}
        <div style={{ marginTop: '2rem' }}>
          <h3 style={{ fontSize: '1rem', color: '#fff', marginBottom: '1rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Award size={20} color="#f59e0b" />
            <span>Phase 3 Baseline Candidate Comparison</span>
          </h3>

          <table className="custom-table">
            <thead>
              <tr>
                <th>Model Candidate</th>
                <th>MAE (Wh)</th>
                <th>RMSE (Wh)</th>
                <th>R² Score</th>
                <th>Selection Status</th>
              </tr>
            </thead>
            <tbody>
              {MODEL_COMPARISON_METRICS.map((row) => (
                <tr key={row.model} style={{ backgroundColor: row.status.includes('Best') ? 'rgba(59, 130, 246, 0.08)' : 'transparent' }}>
                  <td style={{ fontWeight: row.status.includes('Best') ? 700 : 500, color: row.status.includes('Best') ? '#60a5fa' : '#fff' }}>
                    {row.model}
                  </td>
                  <td style={{ fontFamily: 'monospace' }}>{row.mae.toFixed(2)}</td>
                  <td style={{ fontFamily: 'monospace', fontWeight: 700 }}>{row.rmse.toFixed(2)}</td>
                  <td style={{ fontFamily: 'monospace', color: row.r2 > 0 ? '#34d399' : '#f87171' }}>{row.r2.toFixed(4)}</td>
                  <td>
                    <span 
                      style={{ 
                        padding: '0.25rem 0.75rem', 
                        borderRadius: 'var(--radius-full)', 
                        fontSize: '0.75rem', 
                        fontWeight: 700,
                        backgroundColor: row.status.includes('Best') ? 'rgba(16, 185, 129, 0.15)' : 'rgba(255, 255, 255, 0.05)',
                        color: row.status.includes('Best') ? '#34d399' : 'var(--text-muted)',
                        border: row.status.includes('Best') ? '1px solid rgba(16,185,129,0.3)' : '1px solid var(--border-color)'
                      }}
                    >
                      {row.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Model Performance Comparison Chart */}
        <div style={{ marginTop: '2.5rem', background: 'rgba(0,0,0,0.25)', padding: '1.5rem', borderRadius: 'var(--radius-lg)', border: '1px solid var(--border-color)' }}>
          <h4 style={{ fontSize: '1rem', color: '#fff', marginBottom: '1rem', fontWeight: 700 }}>
            Model Evaluation Error Comparison (Lower RMSE/MAE is Better)
          </h4>

          <div style={{ width: '100%', height: '300px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={MODEL_COMPARISON_METRICS} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="model" stroke="#9ca3af" tick={{ fill: '#9ca3af', fontSize: 12 }} />
                <YAxis stroke="#9ca3af" tick={{ fill: '#9ca3af', fontSize: 12 }} />
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', borderColor: 'rgba(255,255,255,0.1)', color: '#fff' }} />
                <Legend wrapperStyle={{ color: '#fff', fontSize: '12px' }} />
                <Bar dataKey="mae" name="MAE (Wh)" fill="#60a5fa" radius={[4, 4, 0, 0]} />
                <Bar dataKey="rmse" name="RMSE (Wh)" fill="#f43f5e" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
