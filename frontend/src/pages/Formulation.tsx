import { useState } from 'react';
import { FlaskConical, Calculator, Search, Loader2 } from 'lucide-react';

const API_BASE = '/api/v1';

interface Candidate {
  components: { material_code: string; name: string; weight_percent: number }[];
  predicted_properties: {
    viscosity_40c: number | null;
    viscosity_100c: number | null;
    viscosity_index: number | null;
    density_15c: number | null;
    flash_point: number | null;
    pour_point: number | null;
  };
  score: number;
  estimated_cost_per_liter: number;
}

export default function Formulation() {
  const [targetKv40, setTargetKv40] = useState('');
  const [targetKv100, setTargetKv100] = useState('');
  const [maxComponents, setMaxComponents] = useState(4);
  const [loading, setLoading] = useState(false);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [error, setError] = useState('');

  const handleFormulate = async () => {
    if (!targetKv40 && !targetKv100) {
      setError('Enter at least one target viscosity');
      return;
    }
    setLoading(true);
    setError('');
    setCandidates([]);

    try {
      const resp = await fetch(`${API_BASE}/formulation/formulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          target_viscosity_40c: targetKv40 ? parseFloat(targetKv40) : null,
          target_viscosity_100c: targetKv100 ? parseFloat(targetKv100) : null,
          max_components: maxComponents,
          iterations: 1000,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || 'Request failed');
      setCandidates(data.candidates || []);
      if (data.candidates?.length === 0) {
        setError('No viable formulations found. Try adjusting targets.');
      }
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">Recipe Formulation</h2>
        <span className="text-sm text-gray-500">AI-Powered Blend Optimizer</span>
      </div>

      {/* Input Panel */}
      <div className="card">
        <h3 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <Calculator className="w-5 h-5 text-blue-600" />
          Target Specifications
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              KV @40°C (cSt)
            </label>
            <input
              type="number"
              value={targetKv40}
              onChange={(e) => setTargetKv40(e.target.value)}
              placeholder="e.g. 140"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              KV @100°C (cSt)
            </label>
            <input
              type="number"
              value={targetKv100}
              onChange={(e) => setTargetKv100(e.target.value)}
              placeholder="e.g. 14.4"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Max Components
            </label>
            <select
              value={maxComponents}
              onChange={(e) => setMaxComponents(parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              {[2, 3, 4, 5, 6].map((n) => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
          </div>
          <div className="flex items-end">
            <button
              onClick={handleFormulate}
              disabled={loading}
              className="btn btn-primary w-full flex items-center justify-center gap-2"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
              {loading ? 'Optimizing...' : 'Find Formulations'}
            </button>
          </div>
        </div>
        {error && (
          <p className="mt-3 text-sm text-red-600">{error}</p>
        )}
      </div>

      {/* Results */}
      {candidates.length > 0 && (
        <div className="space-y-4">
          <h3 className="font-semibold text-gray-900">
            Top {candidates.length} Candidates
          </h3>
          {candidates.map((c, idx) => (
            <div key={idx} className="card hover:shadow-md transition-shadow">
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-2">
                  <span className="bg-blue-100 text-blue-700 text-xs font-bold px-2 py-1 rounded">
                    #{idx + 1}
                  </span>
                  <span className="text-sm font-medium text-gray-700">
                    Score: {c.score} / 100
                  </span>
                </div>
                <span className="text-sm font-medium text-green-700">
                  ${c.estimated_cost_per_liter}/L
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Components */}
                <div>
                  <h4 className="text-xs font-semibold text-gray-500 uppercase mb-2">Components</h4>
                  <div className="space-y-1">
                    {c.components.map((comp, ci) => (
                      <div key={ci} className="flex justify-between text-sm">
                        <span className="text-gray-700">{comp.name}</span>
                        <span className="font-mono font-medium">{comp.weight_percent.toFixed(2)}%</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Predicted Properties */}
                <div>
                  <h4 className="text-xs font-semibold text-gray-500 uppercase mb-2">Predicted Properties</h4>
                  <div className="space-y-1 text-sm">
                    {c.predicted_properties.viscosity_40c && (
                      <div className="flex justify-between">
                        <span className="text-gray-500">KV @40°C</span>
                        <span className="font-medium">{c.predicted_properties.viscosity_40c} cSt</span>
                      </div>
                    )}
                    {c.predicted_properties.viscosity_100c && (
                      <div className="flex justify-between">
                        <span className="text-gray-500">KV @100°C</span>
                        <span className="font-medium">{c.predicted_properties.viscosity_100c} cSt</span>
                      </div>
                    )}
                    {c.predicted_properties.viscosity_index && (
                      <div className="flex justify-between">
                        <span className="text-gray-500">VI</span>
                        <span className="font-medium">{c.predicted_properties.viscosity_index}</span>
                      </div>
                    )}
                    {c.predicted_properties.flash_point && (
                      <div className="flex justify-between">
                        <span className="text-gray-500">Flash Point</span>
                        <span className="font-medium">{c.predicted_properties.flash_point}°C</span>
                      </div>
                    )}
                    {c.predicted_properties.pour_point && (
                      <div className="flex justify-between">
                        <span className="text-gray-500">Pour Point</span>
                        <span className="font-medium">{c.predicted_properties.pour_point}°C</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
