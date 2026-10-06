import React, { useState, useEffect, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import DashboardView from './components/DashboardView';
import PredictionView from './components/PredictionView';
import MonitoringView from './components/MonitoringView';
import ModelInfoView from './components/ModelInfoView';
import SystemHealthView from './components/SystemHealthView';
import { getHealth, getMonitoring } from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [health, setHealth] = useState(null);
  const [monitoring, setMonitoring] = useState(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState(null);

  const fetchBackendStatus = useCallback(async () => {
    setIsRefreshing(true);
    setError(null);
    try {
      const [healthRes, monitoringRes] = await Promise.allSettled([
        getHealth(),
        getMonitoring()
      ]);

      if (healthRes.status === 'fulfilled') {
        setHealth(healthRes.value);
      } else {
        console.error('Health fetch failed:', healthRes.reason);
        setError('Backend service is unreachable on port 8000.');
      }

      if (monitoringRes.status === 'fulfilled') {
        setMonitoring(monitoringRes.value.monitoring);
      } else {
        console.error('Monitoring fetch failed:', monitoringRes.reason);
      }
    } catch (err) {
      setError(err.message || 'Error connecting to FastAPI backend.');
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchBackendStatus();
    // Poll every 30 seconds for live updates
    const interval = setInterval(fetchBackendStatus, 30000);
    return () => clearInterval(interval);
  }, [fetchBackendStatus]);

  const renderActiveView = () => {
    switch (activeTab) {
      case 'dashboard':
        return <DashboardView health={health} monitoring={monitoring} error={error} />;
      case 'prediction':
        return <PredictionView health={health} />;
      case 'monitoring':
        return <MonitoringView monitoring={monitoring} error={error} />;
      case 'model':
        return <ModelInfoView health={health} />;
      case 'health':
        return <SystemHealthView health={health} error={error} onRefresh={fetchBackendStatus} isRefreshing={isRefreshing} />;
      default:
        return <DashboardView health={health} monitoring={monitoring} error={error} />;
    }
  };

  return (
    <div className="app-container">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} health={health} />
      
      <div className="main-wrapper">
        <Header 
          health={health} 
          monitoring={monitoring} 
          onRefresh={fetchBackendStatus} 
          isRefreshing={isRefreshing} 
        />

        <main className="content-body">
          {renderActiveView()}
        </main>
      </div>
    </div>
  );
}
