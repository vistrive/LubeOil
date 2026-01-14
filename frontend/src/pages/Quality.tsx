import { useState } from 'react';
import { CheckCircle, XCircle, AlertTriangle, Brain, TrendingUp } from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

// Mock data
const qualityData = [
  { time: '06:00', viscosity: 94.5, target: 95.5, lower: 93.6, upper: 97.4 },
  { time: '07:00', viscosity: 95.2, target: 95.5, lower: 93.6, upper: 97.4 },
  { time: '08:00', viscosity: 95.8, target: 95.5, lower: 93.6, upper: 97.4 },
  { time: '09:00', viscosity: 95.4, target: 95.5, lower: 93.6, upper: 97.4 },
  { time: '10:00', viscosity: 96.1, target: 95.5, lower: 93.6, upper: 97.4 },
  { time: '11:00', viscosity: 95.7, target: 95.5, lower: 93.6, upper: 97.4 },
];

const recentMeasurements = [
  {
    id: '1',
    batchNumber: 'BL-20240115-A1B2',
    source: 'Inline Analyzer',
    viscosity40c: 95.7,
    flashPoint: 222,
    pourPoint: -28,
    status: 'on_spec',
    time: '10 min ago',
  },
  {
    id: '2',
    batchNumber: 'BL-20240115-C3D4',
    source: 'Lab Analysis',
    viscosity40c: 109.5,
    flashPoint: 226,
    pourPoint: -24,
    status: 'on_spec',
    time: '25 min ago',
  },
  {
    id: '3',
    batchNumber: 'BL-20240114-G7H8',
    source: 'Lab Analysis',
    viscosity40c: 142.3,
    flashPoint: 218,
    pourPoint: -22,
    status: 'marginal',
    time: '2 hours ago',
  },
];

const predictions = [
  {
    id: '1',
    batchNumber: 'BL-20240115-A1B2',
    predictedViscosity: 95.8,
    confidence: 94,
    offSpecRisk: 2.1,
    recommendation: 'No action needed',
  },
  {
    id: '2',
    batchNumber: 'BL-20240115-C3D4',
    predictedViscosity: 110.2,
    confidence: 91,
    offSpecRisk: 3.5,
    recommendation: 'Consider adding 0.2% VI improver',
  },
];

export default function Quality() {
  const [activeTab, setActiveTab] = useState<'measurements' | 'predictions'>('measurements');

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">Quality Control</h2>
        <div className="flex items-center gap-2">
          <span className="flex items-center gap-1 text-green-600 text-sm">
            <CheckCircle className="w-4 h-4" />
            98.5% On-Spec Today
          </span>
        </div>
      </div>

      {/* Quality KPIs */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-green-100 rounded-lg">
              <CheckCircle className="w-5 h-5 text-green-600" />
            </div>
            <div>
              <div className="text-2xl font-bold">156</div>
              <div className="text-sm text-gray-500">On-Spec Batches</div>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-yellow-100 rounded-lg">
              <AlertTriangle className="w-5 h-5 text-yellow-600" />
            </div>
            <div>
              <div className="text-2xl font-bold">3</div>
              <div className="text-sm text-gray-500">Marginal Batches</div>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-red-100 rounded-lg">
              <XCircle className="w-5 h-5 text-red-600" />
            </div>
            <div>
              <div className="text-2xl font-bold">0</div>
              <div className="text-sm text-gray-500">Off-Spec Batches</div>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-purple-100 rounded-lg">
              <Brain className="w-5 h-5 text-purple-600" />
            </div>
            <div>
              <div className="text-2xl font-bold">92%</div>
              <div className="text-sm text-gray-500">AI Prediction Accuracy</div>
            </div>
          </div>
        </div>
      </div>

      {/* Viscosity Trend Chart */}
      <div className="card">
        <h3 className="text-lg font-semibold mb-4">
          Viscosity Trend - BL-20240115-A1B2
        </h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={qualityData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="time" stroke="#6b7280" fontSize={12} />
              <YAxis
                domain={[90, 100]}
                stroke="#6b7280"
                fontSize={12}
                label={{ value: 'cSt', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip />
              <ReferenceLine y={95.5} stroke="#9ca3af" strokeDasharray="5 5" label="Target" />
              <Line
                type="monotone"
                dataKey="viscosity"
                stroke="#2563eb"
                strokeWidth={2}
                dot={{ fill: '#2563eb' }}
                name="Measured"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="flex gap-4">
          <button
            onClick={() => setActiveTab('measurements')}
            className={`py-3 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'measurements'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            Recent Measurements
          </button>
          <button
            onClick={() => setActiveTab('predictions')}
            className={`py-3 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'predictions'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            AI Predictions
          </button>
        </nav>
      </div>

      {/* Measurements Table */}
      {activeTab === 'measurements' && (
        <div className="card overflow-hidden p-0">
          <table className="data-table">
            <thead>
              <tr>
                <th>Batch</th>
                <th>Source</th>
                <th>Viscosity @40°C</th>
                <th>Flash Point</th>
                <th>Pour Point</th>
                <th>Status</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              {recentMeasurements.map((m) => (
                <tr key={m.id}>
                  <td className="font-mono text-sm">{m.batchNumber}</td>
                  <td>{m.source}</td>
                  <td>{m.viscosity40c} cSt</td>
                  <td>{m.flashPoint}°C</td>
                  <td>{m.pourPoint}°C</td>
                  <td>
                    <span
                      className={`flex items-center gap-1 ${
                        m.status === 'on_spec'
                          ? 'text-green-600'
                          : 'text-yellow-600'
                      }`}
                    >
                      {m.status === 'on_spec' ? (
                        <CheckCircle className="w-4 h-4" />
                      ) : (
                        <AlertTriangle className="w-4 h-4" />
                      )}
                      {m.status.replace('_', ' ')}
                    </span>
                  </td>
                  <td className="text-gray-500">{m.time}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Predictions Table */}
      {activeTab === 'predictions' && (
        <div className="card overflow-hidden p-0">
          <table className="data-table">
            <thead>
              <tr>
                <th>Batch</th>
                <th>Predicted Viscosity</th>
                <th>Confidence</th>
                <th>Off-Spec Risk</th>
                <th>AI Recommendation</th>
              </tr>
            </thead>
            <tbody>
              {predictions.map((p) => (
                <tr key={p.id}>
                  <td className="font-mono text-sm">{p.batchNumber}</td>
                  <td>{p.predictedViscosity} cSt</td>
                  <td>
                    <span className="flex items-center gap-1">
                      <TrendingUp className="w-4 h-4 text-green-500" />
                      {p.confidence}%
                    </span>
                  </td>
                  <td>
                    <span
                      className={`font-medium ${
                        p.offSpecRisk > 5 ? 'text-red-600' : 'text-green-600'
                      }`}
                    >
                      {p.offSpecRisk}%
                    </span>
                  </td>
                  <td className="text-sm">{p.recommendation}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
