import React from 'react';
import { ShieldCheck, Activity, Cpu, RefreshCw } from 'lucide-react';

export default function Header({ health, monitoring, onRefresh, isRefreshing }) {
  const isHealthy = health && health.status === 'healthy';
  const hasDrift = monitoring && monitoring.overall_drift_detected;

  return (
    <header className="top-header">
      <div>
        <h1 className="header-title">
          Energy Consumption Prediction
        </h1>
        <p className="header-subtitle">
          MLOps-powered energy forecasting and real-time model drift monitoring
        </p>
      </div>

      <div className="badge-group">
        <div className="badge-item">
          <ShieldCheck size={15} color={isHealthy ? '#34d399' : '#f87171'} />
          <span>{isHealthy ? 'Model Healthy' : 'Model Unhealthy'}</span>
        </div>

        <div className="badge-item">
          <Cpu size={15} color="#60a5fa" />
          <span>Champion</span>
        </div>

        <div className="badge-item">
          <Activity size={15} color={hasDrift ? '#f87171' : '#34d399'} />
          <span>{hasDrift ? 'Drift Detected' : 'No Drift'}</span>
        </div>

        <button 
          className="btn btn-secondary" 
          onClick={onRefresh} 
          disabled={isRefreshing}
          style={{ padding: '0.4rem 0.85rem', fontSize: '0.8rem' }}
        >
          <RefreshCw size={14} className={isRefreshing ? 'spin' : ''} />
          <span>{isRefreshing ? 'Refreshing...' : 'Refresh'}</span>
        </button>
      </div>
    </header>
  );
}
