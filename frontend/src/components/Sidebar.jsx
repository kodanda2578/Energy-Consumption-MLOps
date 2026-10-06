import React from 'react';
import { 
  LayoutDashboard, 
  Zap, 
  Activity, 
  Cpu, 
  ShieldCheck, 
  Server,
  AlertCircle
} from 'lucide-react';

export default function Sidebar({ activeTab, setActiveTab, health }) {
  const isHealthy = health && health.status === 'healthy';

  const menuItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'prediction', label: 'Prediction', icon: Zap },
    { id: 'monitoring', label: 'Data Drift Monitoring', icon: Activity },
    { id: 'model', label: 'Model Information', icon: Cpu },
    { id: 'health', label: 'System Health', icon: ShieldCheck },
  ];

  return (
    <aside className="sidebar">
      <div className="brand-header">
        <div className="brand-icon">
          <Zap size={22} />
        </div>
        <div>
          <div className="brand-title">Energy MLOps</div>
          <div className="brand-subtitle">Forecast & Monitor</div>
        </div>
      </div>

      <ul className="nav-menu">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <li
              key={item.id}
              className={`nav-item ${isActive ? 'active' : ''}`}
              onClick={() => setActiveTab(item.id)}
            >
              <Icon size={18} />
              <span>{item.label}</span>
            </li>
          );
        })}
      </ul>

      <div className="sidebar-footer">
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.5rem', fontWeight: 600 }}>
          FASTAPI BACKEND
        </div>
        <div className={`status-pill ${isHealthy ? 'healthy' : 'unhealthy'}`}>
          <span className="dot"></span>
          <span>{isHealthy ? 'API HEALTHY' : 'API UNREACHABLE'}</span>
        </div>
      </div>
    </aside>
  );
}
