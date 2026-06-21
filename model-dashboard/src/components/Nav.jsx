const PAGES = [
  ["overview", "Overview"],
  ["leaderboard", "Model Leaderboard"],
  ["tracking", "MLflow / Optuna"],
  ["metrics", "Metric Explanation"],
  ["threshold", "Threshold Tuning"],
  ["final", "Final Model"],
  ["predict", "Prediction Demo"],
];

export default function Nav({ page, setPage }) {
  return (
    <nav className="sidebar">
      <div className="brand">
        Claim Approval
        <small>Model Optimization Dashboard</small>
      </div>
      {PAGES.map(([key, label]) => (
        <button
          key={key}
          className={"nav-item" + (page === key ? " active" : "")}
          onClick={() => setPage(key)}
        >
          {label}
        </button>
      ))}
    </nav>
  );
}
