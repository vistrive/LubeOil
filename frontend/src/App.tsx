import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Recipes from './pages/Recipes';
import Tanks from './pages/Tanks';
import Blends from './pages/Blends';
import Quality from './pages/Quality';
import Formulation from './pages/Formulation';
import AIChat from './pages/AIChat';

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/recipes" element={<Recipes />} />
        <Route path="/formulation" element={<Formulation />} />
        <Route path="/tanks" element={<Tanks />} />
        <Route path="/blends" element={<Blends />} />
        <Route path="/quality" element={<Quality />} />
        <Route path="/ai-chat" element={<AIChat />} />
      </Routes>
    </Layout>
  );
}

export default App;
