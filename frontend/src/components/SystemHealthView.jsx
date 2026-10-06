import React from 'react';
import { ShieldCheck, Server, Cpu, Tag, CheckCircle2, XCircle, AlertTriangle, RefreshCw } from 'lucide-react';

export default function SystemHealthView({ health, error, onRefresh, isRefreshing }) {
  const isHealthy = health && health.status === 'healthy';

  const endpoints = [
    { path: '/', method: 'GET', description: 'Root project status and API overview link', status: 'Online' },
    { path: '/health', method: 'GET', description: 'System health check and model loading state', status: isHealthy ? 'Healthy' : 'Error' },
    { path: '/monitoring', method: 'GET', description: 'Data drift summary report and Kolmogorov-Smirnov metrics', status: 'Active' },
    { path: '/predict', method: 'POST', description: 'Inference endpoint taking 41 numerical features', status: health?.model_loaded ? 'Ready' : 'Unavailable' },
  ];

  return (
    <div className="system-health-view">
      <div className="form-section">
        <div className="section-header" style={{ justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <ShieldCheck size={24} color={isHealthy ? '#34d399' : '#f87171'} />
            <div>
              <h2 className="section-title">System & API Health Diagnostics</h2>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                Real-time operational status fetched from FastAPI <code style={{ color: '#93c5fd' }}>GET /health</code>
              </p>
            </div>
          </div>

          <button type="button" className="btn btn-secondary" onClick={onRefresh} disabled={isRefreshing}>
            <RefreshCw size={16} className={isRefreshing ? 'spin' : ''} />
            <span>{isRefreshing ? 'Checking...' : 'Check API Status'}</span>
          </button>
        </div>

        {error && (
          <div className="error-banner">
            <AlertTriangle size={20} />
            <span>API Health Check Failed: {error}. Make sure FastAPI is listening on http://localhost:8000.</span>
          </div>
        )}

        {/* Top Operational Status Cards */}
        <div className="card-grid">
          <div className="card">
            <div className="card-header">
              <span className="card-title">FastAPI Service</span>
              <Server size={20} color={isHealthy ? '#34d399' : '#f87171'} />
            </div>
            <div className="card-value" style={{ color: isHealthy ? '#34d399' : '#f87171' }}>
              {isHealthy ? 'Healthy' : 'Unhealthy'}
            </div>
            <div className="card-subtext">Port 8000 REST Service</div>
          </div>

          <div className="card">
            <div className="card-header">
              <span className="card-title">MLflow Model Load</span>
              <Cpu size={20} color={health?.model_loaded ? '#34d399' : '#f87171'} />
            </div>
            <div className="card-value" style={{ color: health?.model_loaded ? '#34d399' : '#f87171' }}>
              {health?.model_loaded ? 'Loaded' : 'Unloaded'}
            </div>
            <div className="card-subtext">PyFunc Model Memory State</div>
          </div>

          <div className="card">
            <div className="card-header">
              <span className="card-title">Registry Model</span>
              <Cpu size={20} color="#60a5fa" />
            </div>
            <div className="card-value" style={{ fontSize: '1.2rem', color: '#93c5fd' }}>
              {health?.model_name || 'EnergyConsumptionModel'}
            </div>
            <div className="card-subtext">MLflow Experiment Registry</div>
          </div>

          <div className="card">
            <div className="card-header">
              <span className="card-title">Version Alias</span>
              <Tag size={20} color="#c084fc" />
            </div>
            <div className="card-value" style={{ color: '#c084fc' }}>
              {health?.model_alias || 'champion'}
            </div>
            <div className="card-subtext">Champion Alias Tag</div>
          </div>
        </div>

        {/* Endpoints Table */}
        <div style={{ marginTop: '2rem' }}>
          <h3 style={{ fontSize: '1rem', color: '#fff', marginBottom: '1rem', fontWeight: 700 }}>
            FastAPI REST Endpoints Health Table
          </h3>

          <table className="custom-table">
            <thead>
              <tr>
                <th>Endpoint Path</th>
                <th>HTTP Method</th>
                <th>Description</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {endpoints.map((ep) => (
                <tr key={ep.path}>
                  <td style={{ fontFamily: 'monospace', fontWeight: 700, color: '#93c5fd' }}>{ep.path}</td>
                  <td>
                    <span style={{ padding: '0.2rem 0.5rem', borderRadius: '4px', background: 'rgba(255,255,255,0.05)', fontFamily: 'monospace', fontSize: '0.8rem', fontWeight: 700 }}>
                      {ep.method}
                    </span>
                  </td>
                  <td style={{ color: 'var(--text-muted)' }}>{ep.description}</td>
                  <td>
                    <span 
                      style={{ 
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '0.35rem',
                        padding: '0.25rem 0.75rem', 
                        borderRadius: 'var(--radius-full)', 
                        fontSize: '0.75rem', 
                        fontWeight: 700,
                        backgroundColor: 'rgba(16, 185, 129, 0.15)',
                        color: '#34d399',
                        border: '1px solid rgba(16,185,129,0.3)'
                      }}
                    >
                      <CheckCircle2 size={13} />
                      <span>{ep.status}</span>
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
