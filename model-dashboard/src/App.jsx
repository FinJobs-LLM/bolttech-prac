import { useEffect, useState } from "react";
import Nav from "./components/Nav.jsx";
import { loadDashboard } from "./api/client.js";
import Overview from "./pages/Overview.jsx";
import Leaderboard from "./pages/Leaderboard.jsx";
import Tracking from "./pages/Tracking.jsx";
import MetricExplanation from "./pages/MetricExplanation.jsx";
import Threshold from "./pages/Threshold.jsx";
import FinalModel from "./pages/FinalModel.jsx";
import PredictionDemo from "./pages/PredictionDemo.jsx";

export default function App() {
  const [page, setPage] = useState("overview");
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadDashboard().then(setData).catch((e) => setError(e.message));
  }, []);

  if (error) {
    return (
      <div className="app">
        <Nav page={page} setPage={setPage} />
        <div className="content">
          <h1>Dashboard data not found</h1>
          <div className="warn-box">
            {error}. Run the training pipeline first:
            <pre>python src/run_pipeline.py</pre>
            It writes <code>reports/dashboard_data.json</code> and copies it to{" "}
            <code>model-dashboard/public/</code>.
          </div>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="app">
        <Nav page={page} setPage={setPage} />
        <div className="content"><div className="loading">Loading dashboard…</div></div>
      </div>
    );
  }

  const pages = {
    overview: <Overview data={data} />,
    leaderboard: <Leaderboard data={data} />,
    tracking: <Tracking data={data} />,
    metrics: <MetricExplanation data={data} />,
    threshold: <Threshold data={data} />,
    final: <FinalModel data={data} />,
    predict: <PredictionDemo data={data} />,
  };

  return (
    <div className="app">
      <Nav page={page} setPage={setPage} />
      <div className="content">{pages[page]}</div>
    </div>
  );
}
