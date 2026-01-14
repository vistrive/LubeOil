import { useState } from 'react';
import { Container, Thermometer, Droplets, AlertTriangle } from 'lucide-react';

// Mock data
const tanks = [
  {
    id: '1',
    tag: 'T-101',
    name: 'Base Oil Storage 1',
    type: 'base_oil',
    material: 'SN-150',
    capacity: 50000,
    level: 42500,
    levelPercent: 85,
    temperature: 45,
    status: 'available',
  },
  {
    id: '2',
    tag: 'T-102',
    name: 'Base Oil Storage 2',
    type: 'base_oil',
    material: 'SN-500',
    capacity: 50000,
    level: 15000,
    levelPercent: 30,
    temperature: 42,
    status: 'available',
  },
  {
    id: '3',
    tag: 'T-201',
    name: 'Additive Tank 1',
    type: 'additive',
    material: 'VI Improver',
    capacity: 10000,
    level: 8500,
    levelPercent: 85,
    temperature: 25,
    status: 'in_use',
  },
  {
    id: '4',
    tag: 'BT-01',
    name: 'Blend Tank 1',
    type: 'blend',
    material: 'SAE 10W-40 (In Progress)',
    capacity: 20000,
    level: 12000,
    levelPercent: 60,
    temperature: 55,
    status: 'in_use',
  },
  {
    id: '5',
    tag: 'T-301',
    name: 'Finished Product 1',
    type: 'finished',
    material: 'SAE 15W-40',
    capacity: 100000,
    level: 75000,
    levelPercent: 75,
    temperature: 30,
    status: 'available',
  },
];

function TankLevelIndicator({ percent, status }: { percent: number; status: string }) {
  const getColor = () => {
    if (percent >= 90) return 'bg-red-400';
    if (percent <= 20) return 'bg-yellow-400';
    return 'bg-blue-400';
  };

  return (
    <div className="tank-level">
      <div
        className={`tank-level-fill ${getColor()}`}
        style={{ height: `${percent}%` }}
      />
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-xs font-bold text-gray-800">{percent}%</span>
      </div>
    </div>
  );
}

export default function Tanks() {
  const [selectedType, setSelectedType] = useState<string>('all');

  const tankTypes = [
    { id: 'all', label: 'All Tanks' },
    { id: 'base_oil', label: 'Base Oil' },
    { id: 'additive', label: 'Additives' },
    { id: 'blend', label: 'Blend Tanks' },
    { id: 'finished', label: 'Finished Products' },
  ];

  const filteredTanks = selectedType === 'all'
    ? tanks
    : tanks.filter(t => t.type === selectedType);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">Tank Farm Overview</h2>
        <div className="text-sm text-gray-500">
          {tanks.length} tanks | {tanks.filter(t => t.status === 'in_use').length} in use
        </div>
      </div>

      {/* Type Filter */}
      <div className="flex gap-2">
        {tankTypes.map((type) => (
          <button
            key={type.id}
            onClick={() => setSelectedType(type.id)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              selectedType === type.id
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            {type.label}
          </button>
        ))}
      </div>

      {/* Tank Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {filteredTanks.map((tank) => (
          <div key={tank.id} className="card hover:shadow-md transition-shadow">
            <div className="flex items-start justify-between mb-4">
              <div>
                <span className="text-lg font-bold text-gray-900">{tank.tag}</span>
                <p className="text-sm text-gray-500">{tank.name}</p>
              </div>
              <span
                className={`px-2 py-1 rounded text-xs font-medium ${
                  tank.status === 'in_use'
                    ? 'bg-blue-100 text-blue-700'
                    : 'bg-green-100 text-green-700'
                }`}
              >
                {tank.status.replace('_', ' ')}
              </span>
            </div>

            <div className="flex gap-4">
              <TankLevelIndicator percent={tank.levelPercent} status={tank.status} />

              <div className="flex-1 space-y-2 text-sm">
                <div>
                  <span className="text-gray-500">Material</span>
                  <p className="font-medium truncate">{tank.material}</p>
                </div>
                <div className="flex items-center gap-1 text-gray-600">
                  <Droplets className="w-3 h-3" />
                  <span>{tank.level.toLocaleString()} / {tank.capacity.toLocaleString()} L</span>
                </div>
                <div className="flex items-center gap-1 text-gray-600">
                  <Thermometer className="w-3 h-3" />
                  <span>{tank.temperature}°C</span>
                </div>
              </div>
            </div>

            {tank.levelPercent >= 90 && (
              <div className="mt-4 p-2 bg-yellow-50 rounded flex items-center gap-2 text-yellow-700 text-sm">
                <AlertTriangle className="w-4 h-4" />
                High level warning
              </div>
            )}

            {tank.levelPercent <= 20 && (
              <div className="mt-4 p-2 bg-red-50 rounded flex items-center gap-2 text-red-700 text-sm">
                <AlertTriangle className="w-4 h-4" />
                Low level alert
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
