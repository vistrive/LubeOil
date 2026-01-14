import { useState } from 'react';
import { Plus, Search, Filter, Edit, Trash2, CheckCircle } from 'lucide-react';

// Mock data
const recipes = [
  {
    id: '1',
    code: 'SAE-10W40-001',
    name: 'SAE 10W-40 Motor Oil',
    status: 'approved',
    viscosity40c: 95.5,
    flashPoint: 220,
    pourPoint: -30,
    ingredients: 5,
  },
  {
    id: '2',
    code: 'SAE-15W40-002',
    name: 'SAE 15W-40 Heavy Duty',
    status: 'approved',
    viscosity40c: 110.0,
    flashPoint: 225,
    pourPoint: -25,
    ingredients: 6,
  },
  {
    id: '3',
    code: 'ATF-DEX3-001',
    name: 'ATF Dexron III',
    status: 'draft',
    viscosity40c: 35.5,
    flashPoint: 180,
    pourPoint: -45,
    ingredients: 8,
  },
];

export default function Recipes() {
  const [searchQuery, setSearchQuery] = useState('');

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">Recipe Management</h2>
        <button className="btn btn-primary">
          <Plus className="w-4 h-4" />
          New Recipe
        </button>
      </div>

      {/* Search and Filter */}
      <div className="flex gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search recipes..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>
        <button className="btn btn-secondary">
          <Filter className="w-4 h-4" />
          Filters
        </button>
      </div>

      {/* Recipe Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {recipes.map((recipe) => (
          <div key={recipe.id} className="card hover:shadow-md transition-shadow">
            <div className="flex items-start justify-between mb-4">
              <div>
                <span className="text-xs font-mono text-gray-500">{recipe.code}</span>
                <h3 className="font-semibold text-gray-900 mt-1">{recipe.name}</h3>
              </div>
              <span
                className={`px-2 py-1 rounded text-xs font-medium ${
                  recipe.status === 'approved'
                    ? 'bg-green-100 text-green-700'
                    : 'bg-yellow-100 text-yellow-700'
                }`}
              >
                {recipe.status}
              </span>
            </div>

            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-500">Viscosity @40°C</span>
                <span className="font-medium">{recipe.viscosity40c} cSt</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Flash Point</span>
                <span className="font-medium">{recipe.flashPoint}°C</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Pour Point</span>
                <span className="font-medium">{recipe.pourPoint}°C</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Ingredients</span>
                <span className="font-medium">{recipe.ingredients}</span>
              </div>
            </div>

            <div className="flex gap-2 mt-4 pt-4 border-t border-gray-100">
              <button className="btn btn-secondary flex-1 text-sm py-1.5">
                <Edit className="w-3 h-3" />
                Edit
              </button>
              {recipe.status === 'draft' && (
                <button className="btn btn-success flex-1 text-sm py-1.5">
                  <CheckCircle className="w-3 h-3" />
                  Approve
                </button>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
