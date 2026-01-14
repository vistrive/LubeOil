import {
  Activity,
  Droplets,
  TrendingUp,
  AlertTriangle,
  Zap,
  CheckCircle,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
} from 'recharts';

// Mock data for demonstration
const productionData = [
  { time: '06:00', volume: 4200, target: 4000 },
  { time: '08:00', volume: 8100, target: 8000 },
  { time: '10:00', volume: 12500, target: 12000 },
  { time: '12:00', volume: 15800, target: 16000 },
  { time: '14:00', volume: 19200, target: 20000 },
  { time: '16:00', volume: 23100, target: 24000 },
];

const blendQueue = [
  { id: 'BL-20240115-A1B2', recipe: 'SAE 10W-40', volume: 5000, status: 'in_progress', progress: 65 },
  { id: 'BL-20240115-C3D4', recipe: 'SAE 15W-40', volume: 3500, status: 'mixing', progress: 30 },
  { id: 'BL-20240115-E5F6', recipe: 'ATF Dexron III', volume: 2000, status: 'queued', progress: 0 },
];

const activeAlarms = [
  { id: 1, priority: 'high', message: 'Tank T-101 approaching high level (88%)', time: '10 min ago' },
  { id: 2, priority: 'medium', message: 'Pump P-203 vibration above threshold', time: '25 min ago' },
];

export default function Dashboard() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">Operations Dashboard</h2>
        <div className="text-sm text-gray-500">
          Last updated: {new Date().toLocaleTimeString()}
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="kpi-card">
          <div className="flex items-center justify-between">
            <div>
              <div className="kpi-value">23,100</div>
              <div className="kpi-label">Liters Produced Today</div>
            </div>
            <div className="p-3 bg-blue-100 rounded-lg">
              <Droplets className="w-6 h-6 text-blue-600" />
            </div>
          </div>
          <div className="mt-4 flex items-center text-sm">
            <TrendingUp className="w-4 h-4 text-green-500 mr-1" />
            <span className="text-green-600">+8.5%</span>
            <span className="text-gray-500 ml-2">vs yesterday</span>
          </div>
        </div>

        <div className="kpi-card">
          <div className="flex items-center justify-between">
            <div>
              <div className="kpi-value">2</div>
              <div className="kpi-label">Active Blends</div>
            </div>
            <div className="p-3 bg-green-100 rounded-lg">
              <Activity className="w-6 h-6 text-green-600" />
            </div>
          </div>
          <div className="mt-4 flex items-center text-sm">
            <span className="text-gray-500">3 in queue</span>
          </div>
        </div>

        <div className="kpi-card">
          <div className="flex items-center justify-between">
            <div>
              <div className="kpi-value">98.5%</div>
              <div className="kpi-label">Quality On-Spec Rate</div>
            </div>
            <div className="p-3 bg-purple-100 rounded-lg">
              <CheckCircle className="w-6 h-6 text-purple-600" />
            </div>
          </div>
          <div className="mt-4 flex items-center text-sm">
            <TrendingUp className="w-4 h-4 text-green-500 mr-1" />
            <span className="text-green-600">+2.1%</span>
            <span className="text-gray-500 ml-2">this week</span>
          </div>
        </div>

        <div className="kpi-card">
          <div className="flex items-center justify-between">
            <div>
              <div className="kpi-value">4.2</div>
              <div className="kpi-label">kWh per 1000L</div>
            </div>
            <div className="p-3 bg-yellow-100 rounded-lg">
              <Zap className="w-6 h-6 text-yellow-600" />
            </div>
          </div>
          <div className="mt-4 flex items-center text-sm">
            <TrendingUp className="w-4 h-4 text-green-500 mr-1" />
            <span className="text-green-600">-12%</span>
            <span className="text-gray-500 ml-2">energy savings</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Production Chart */}
        <div className="lg:col-span-2 card">
          <h3 className="text-lg font-semibold mb-4">Production Volume</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={productionData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="time" stroke="#6b7280" fontSize={12} />
                <YAxis stroke="#6b7280" fontSize={12} />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="volume"
                  stroke="#2563eb"
                  strokeWidth={2}
                  dot={false}
                  name="Actual"
                />
                <Line
                  type="monotone"
                  dataKey="target"
                  stroke="#9ca3af"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  dot={false}
                  name="Target"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Active Alarms */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Active Alarms</h3>
            <span className="px-2 py-1 bg-red-100 text-red-700 text-xs font-medium rounded">
              {activeAlarms.length} Active
            </span>
          </div>
          <div className="space-y-3">
            {activeAlarms.map((alarm) => (
              <div
                key={alarm.id}
                className={`p-3 rounded-lg ${
                  alarm.priority === 'high' ? 'alarm-high' : 'alarm-medium'
                }`}
              >
                <div className="flex items-start gap-2">
                  <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="text-sm font-medium">{alarm.message}</p>
                    <p className="text-xs opacity-75 mt-1">{alarm.time}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Blend Queue */}
      <div className="card">
        <h3 className="text-lg font-semibold mb-4">Blend Queue</h3>
        <table className="data-table">
          <thead>
            <tr>
              <th>Batch Number</th>
              <th>Recipe</th>
              <th>Volume (L)</th>
              <th>Status</th>
              <th>Progress</th>
            </tr>
          </thead>
          <tbody>
            {blendQueue.map((blend) => (
              <tr key={blend.id}>
                <td className="font-mono text-sm">{blend.id}</td>
                <td>{blend.recipe}</td>
                <td>{blend.volume.toLocaleString()}</td>
                <td>
                  <span
                    className={`px-2 py-1 rounded text-xs font-medium ${
                      blend.status === 'in_progress'
                        ? 'bg-green-100 text-green-700'
                        : blend.status === 'mixing'
                        ? 'bg-blue-100 text-blue-700'
                        : 'bg-gray-100 text-gray-700'
                    }`}
                  >
                    {blend.status.replace('_', ' ')}
                  </span>
                </td>
                <td>
                  <div className="flex items-center gap-2">
                    <div className="progress-bar w-24">
                      <div
                        className="progress-bar-fill"
                        style={{ width: `${blend.progress}%` }}
                      />
                    </div>
                    <span className="text-sm text-gray-500">{blend.progress}%</span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
