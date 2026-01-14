import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Recipes from './pages/Recipes';
import Tanks from './pages/Tanks';
import Blends from './pages/Blends';
import Quality from './pages/Quality';

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/recipes" element={<Recipes />} />
        <Route path="/tanks" element={<Tanks />} />
        <Route path="/blends" element={<Blends />} />
        <Route path="/quality" element={<Quality />} />
      </Routes>
    </Layout>
  );
}

export default App;
