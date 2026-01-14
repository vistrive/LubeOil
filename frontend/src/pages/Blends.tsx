import { useState } from 'react';
import { Plus, Play, Pause, StopCircle, Eye, Brain } from 'lucide-react';

// Mock data
const blends = [
  {
    id: '1',
    batchNumber: 'BL-20240115-A1B2',
    recipe: 'SAE 10W-40',
    targetVolume: 5000,
    actualVolume: 3250,
    status: 'in_progress',
    progress: 65,
    blendTank: 'BT-01',
    aiOptimized: true,
    offSpecRisk: 2.1,
    startTime: '2024-01-15T06:30:00',
  },
  {
    id: '2',
    batchNumber: 'BL-20240115-C3D4',
    recipe: 'SAE 15W-40',
    targetVolume: 3500,
    actualVolume: 1050,
    status: 'mixing',
    progress: 30,
    blendTank: 'BT-02',
    aiOptimized: true,
    offSpecRisk: 1.5,
    startTime: '2024-01-15T08:00:00',
  },
  {
    id: '3',
    batchNumber: 'BL-20240115-E5F6',
    recipe: 'ATF Dexron III',
    targetVolume: 2000,
    actualVolume: 0,
    status: 'queued',
    progress: 0,
    blendTank: '-',
    aiOptimized: false,
    offSpecRisk: 0,
    startTime: null,
  },
  {
    id: '4',
    batchNumber: 'BL-20240114-G7H8',
    recipe: 'SAE 20W-50',
    targetVolume: 8000,
    actualVolume: 8000,
    status: 'completed',
    progress: 100,
    blendTank: 'BT-01',
    aiOptimized: true,
    offSpecRisk: 0,
    startTime: '2024-01-14T14:00:00',
  },
];

const statusColors: Record<string, string> = {
  in_progress: 'bg-green-100 text-green-700',
  mixing: 'bg-blue-100 text-blue-700',
  queued: 'bg-gray-100 text-gray-700',
  completed: 'bg-purple-100 text-purple-700',
  on_hold: 'bg-yellow-100 text-yellow-700',
};

export default function Blends() {
  const [selectedStatus, setSelectedStatus] = useState<string>('all');

  const statuses = ['all', 'in_progress', 'mixing', 'queued', 'completed'];

  const filteredBlends = selectedStatus === 'all'
    ? blends
    : blends.filter(b => b.status === selectedStatus);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">Blend Operations</h2>
        <button className="btn btn-primary">
          <Plus className="w-4 h-4" />
          New Blend
        </button>
      </div>

      {/* Status Filter */}
      <div className="flex gap-2">
        {statuses.map((status) => (
          <button
            key={status}
            onClick={() => setSelectedStatus(status)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              selectedStatus === status
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            {status === 'all' ? 'All' : status.replace('_', ' ')}
          </button>
        ))}
      </div>

      {/* Blend Table */}
      <div className="card overflow-hidden p-0">
        <table className="data-table">
          <thead>
            <tr>
              <th>Batch Number</th>
              <th>Recipe</th>
              <th>Volume</th>
              <th>Tank</th>
              <th>Status</th>
              <th>Progress</th>
              <th>AI</th>
              <th>Risk</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filteredBlends.map((blend) => (
              <tr key={blend.id}>
                <td className="font-mono text-sm">{blend.batchNumber}</td>
                <td>{blend.recipe}</td>
                <td>
                  {blend.actualVolume.toLocaleString()} / {blend.targetVolume.toLocaleString()} L
                </td>
                <td>{blend.blendTank}</td>
                <td>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${statusColors[blend.status]}`}>
                    {blend.status.replace('_', ' ')}
                  </span>
                </td>
                <td>
                  <div className="flex items-center gap-2">
                    <div className="progress-bar w-20">
                      <div
                        className="progress-bar-fill"
                        style={{ width: `${blend.progress}%` }}
                      />
                    </div>
                    <span className="text-sm text-gray-500">{blend.progress}%</span>
                  </div>
                </td>
                <td>
                  {blend.aiOptimized && (
                    <span className="flex items-center gap-1 text-purple-600">
                      <Brain className="w-4 h-4" />
                    </span>
                  )}
                </td>
                <td>
                  {blend.offSpecRisk > 0 && (
                    <span className={`text-sm font-medium ${
                      blend.offSpecRisk > 5 ? 'text-red-600' : 'text-green-600'
                    }`}>
                      {blend.offSpecRisk}%
                    </span>
                  )}
                </td>
                <td>
                  <div className="flex gap-1">
                    <button className="p-1.5 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded">
                      <Eye className="w-4 h-4" />
                    </button>
                    {blend.status === 'queued' && (
                      <button className="p-1.5 text-green-500 hover:text-green-700 hover:bg-green-50 rounded">
                        <Play className="w-4 h-4" />
                      </button>
                    )}
                    {blend.status === 'in_progress' && (
                      <>
                        <button className="p-1.5 text-yellow-500 hover:text-yellow-700 hover:bg-yellow-50 rounded">
                          <Pause className="w-4 h-4" />
                        </button>
                        <button className="p-1.5 text-red-500 hover:text-red-700 hover:bg-red-50 rounded">
                          <StopCircle className="w-4 h-4" />
                        </button>
                      </>
                    )}
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
