import React from 'react';
import { Activity, ShieldCheck, AlertTriangle, Clock, Percent, ListFilter, CheckCircle2 } from 'lucide-react';
import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip } from 'recharts';

export default function MonitoringView({ monitoring, error }) {
  const hasDrift = monitoring && monitoring.overall_drift_detected;

  const totalFeatures = monitoring?.total_features ?? 41;
  const driftedFeatures = monitoring?.drifted_features ?? 0;
  const cleanFeatures = totalFeatures - driftedFeatures;
  const driftPercentage = monitoring?.drift_percentage ?? 0.0;
  const alpha = monitoring?.alpha ?? 0.05;
  const timestamp = monitoring?.timestamp ?? new Date().toISOString();

  const chartData = [
    { name: 'Stable Features', value: cleanFeatures, color: '#34d399' },
    { name: 'Drifted Features', value: driftedFeatures, color: '#f87171' },
  ];

  return (
    <div className="monitoring-view">
      <div className="form-section">
        <div className="section-header">
          <Activity size={24} color={hasDrift ? '#f87171' : '#34d399'} />
          <div>
            <h2 className="section-title">Data Drift Monitoring Dashboard</h2>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Statistical Kolmogorov-Smirnov (KS) drift testing against reference baseline
            </p>
          </div>
        </div>

        {error && (
          <div className="error-banner">
            <AlertTriangle size={20} />
            <span>Failed to fetch monitoring metrics from GET /monitoring: {error}</span>
          </div>
        )}

        {/* Status Indicator Banner */}
        <div 
          style={{
            background: hasDrift ? 'rgba(244, 63, 94, 0.15)' : 'rgba(16, 185, 129, 0.15)',
            border: `1px solid ${hasDrift ? 'rgba(244, 63, 94, 0.4)' : 'rgba(16, 185, 129, 0.4)'}`,
            borderRadius: 'var(--radius-md)',
            padding: '1.25rem 1.5rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: '2rem'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            {hasDrift ? <AlertTriangle size={32} color="#f87171" /> : <CheckCircle2 size={32} color="#34d399" />}
            <div>
              <div style={{ fontSize: '1.2rem', fontWeight: 800, color: hasDrift ? '#f87171' : '#34d399' }}>
                {hasDrift ? '⚠ DATA DRIFT DETECTED' : '✔ NO DATA DRIFT DETECTED'}
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                Feature distributions conform to Kolmogorov-Smirnov test bounds at α = {alpha}
              </div>
            </div>
          </div>

          <div className={`status-pill ${hasDrift ? 'unhealthy' : 'healthy'}`} style={{ padding: '0.5rem 1.25rem', fontSize: '0.9rem' }}>
            {hasDrift ? 'STATUS: DRIFTED' : 'STATUS: NORMAL'}
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="card-grid">
          <div className="card">
            <div className="card-header">
              <span className="card-title">Total Features</span>
              <ListFilter size={20} color="#60a5fa" />
            </div>
            <div className="card-value">{totalFeatures}</div>
            <div className="card-subtext">Evaluated Numerical Features</div>
          </div>

          <div className="card">
            <div className="card-header">
              <span className="card-title">Drifted Features</span>
              <AlertTriangle size={20} color={driftedFeatures > 0 ? '#f87171' : '#34d399'} />
            </div>
            <div className="card-value" style={{ color: driftedFeatures > 0 ? '#f87171' : '#34d399' }}>
              {driftedFeatures}
            </div>
            <div className="card-subtext">Statistically Drifted Count</div>
          </div>

          <div className="card">
            <div className="card-header">
              <span className="card-title">Drift Percentage</span>
              <Percent size={20} color="#c084fc" />
            </div>
            <div className="card-value">{driftPercentage}%</div>
            <div className="card-subtext">Proportion of Total Pipeline</div>
          </div>

          <div className="card">
            <div className="card-header">
              <span className="card-title">Alpha Threshold</span>
              <ShieldCheck size={20} color="#38bdf8" />
            </div>
            <div className="card-value">α = {alpha}</div>
            <div className="card-subtext">KS Significance Cutoff</div>
          </div>
        </div>

        {/* Chart & Detail Section */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '2rem', marginTop: '1rem' }}>
          {/* Visual Pie Chart */}
          <div style={{ background: 'rgba(0,0,0,0.25)', padding: '1.5rem', borderRadius: 'var(--radius-lg)', border: '1px solid var(--border-color)' }}>
            <h4 style={{ fontSize: '1rem', color: '#fff', marginBottom: '1rem', fontWeight: 700 }}>
              Feature Distribution Health Ratio
            </h4>

            <div style={{ width: '100%', height: '220px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={chartData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={85}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1f2937', borderColor: 'rgba(255,255,255,0.1)', color: '#fff' }} 
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div style={{ display: 'flex', justifyContent: 'center', gap: '2rem', marginTop: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem' }}>
                <span style={{ width: '12px', height: '12px', backgroundColor: '#34d399', borderRadius: '3px' }}></span>
                <span>Stable ({cleanFeatures})</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem' }}>
                <span style={{ width: '12px', height: '12px', backgroundColor: '#f87171', borderRadius: '3px' }}></span>
                <span>Drifted ({driftedFeatures})</span>
              </div>
            </div>
          </div>

          {/* Audit Timestamp & Reference Log */}
          <div style={{ background: 'rgba(0,0,0,0.25)', padding: '1.5rem', borderRadius: 'var(--radius-lg)', border: '1px solid var(--border-color)', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            <div>
              <h4 style={{ fontSize: '1rem', color: '#fff', marginBottom: '1rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Clock size={18} color="#93c5fd" />
                <span>Monitoring Audit Log</span>
              </h4>

              <table className="custom-table">
                <tbody>
                  <tr>
                    <td>Evaluation Method</td>
                    <td style={{ fontWeight: 600 }}>Two-Sample Kolmogorov-Smirnov</td>
                  </tr>
                  <tr>
                    <td>Reference Dataset</td>
                    <td style={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>data/processed/energydata_processed.csv</td>
                  </tr>
                  <tr>
                    <td>MLflow Tracking</td>
                    <td style={{ color: '#34d399', fontWeight: 600 }}>Logged under "Energy Consumption Prediction"</td>
                  </tr>
                  <tr>
                    <td>Last Evaluated Timestamp</td>
                    <td style={{ fontFamily: 'monospace', fontSize: '0.8rem', color: '#93c5fd' }}>{timestamp}</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginTop: '1.5rem', fontStyle: 'italic' }}>
              Note: Data drift monitoring runs automatically in CI/CD and on demand via GET /monitoring.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
