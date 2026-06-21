import DataTable from "../components/DataTable.jsx";

const pct = (v) => (v == null ? "—" : (v * 100).toFixed(1) + "%");
const num = (v) => (v == null ? "—" : Number(v).toFixed(3));

export default function Leaderboard({ data }) {
  const rows = data.leaderboard.map((r) => ({
    ...r,
    modelLabel: r.model,
    stageLabel: r.stage,
  }));

  const columns = [
    { key: "modelLabel", label: "Model", fmt: (v, r) =>
        <span>{v} {r.is_best && <span className="badge best">BEST</span>}</span> },
    { key: "stageLabel", label: "Type", fmt: (v) =>
        <span className={"badge " + (v === "optimized" ? "opt" : "base")}>{v}</span> },
    { key: "imbalance_strategy", label: "Imbalance strategy", fmt: (v) => <span className="note">{v}</span> },
    { key: "val_pr_auc", label: "Val PR-AUC", fmt: (v) => <span className="metric-strong">{num(v)}</span> },
    { key: "test_pr_auc", label: "Test PR-AUC", fmt: (v) => <span className="metric-strong">{num(v)}</span> },
    { key: "test_f1_declined", label: "F1 (Declined)", fmt: num },
    { key: "test_recall_declined", label: "Recall (Declined)", fmt: pct },
    { key: "test_precision_declined", label: "Precision (Declined)", fmt: pct },
    { key: "test_balanced_accuracy", label: "Balanced Acc", fmt: pct },
    { key: "test_accuracy", label: "Accuracy", fmt: pct, deEmph: true },
    { key: "threshold", label: "Threshold", fmt: num },
  ];

  return (
    <div>
      <h1>Model Leaderboard</h1>
      <p className="subtitle">
        All baseline and Optuna-optimized models, ranked by validation PR-AUC. Accuracy is shown
        but greyed out — it is misleading on imbalanced data.
      </p>
      <div className="panel" style={{ overflowX: "auto" }}>
        <DataTable columns={columns} rows={rows} isBest={(r) => r.is_best} />
      </div>
      <p className="note">
        Sorting: click any column header. The highlighted row is the selected best model
        (highest validation PR-AUC). Metrics are reported on the held-out test set unless prefixed “Val”.
      </p>
    </div>
  );
}
