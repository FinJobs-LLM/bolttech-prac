import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer,
  Tooltip, XAxis, YAxis, Legend,
} from "recharts";

export default function Threshold({ data }) {
  const ta = data.threshold_analysis;
  const selected = data.best_model.threshold;
  const cm = data.best_model.confusion_matrix_test; // [[tn,fp],[fn,tp]]

  const chartData = ta.map((r) => ({
    threshold: r.threshold,
    Precision: +r.precision_declined.toFixed(3),
    Recall: +r.recall_declined.toFixed(3),
    F1: +r.f1_declined.toFixed(3),
    "Balanced Acc": +r.balanced_accuracy.toFixed(3),
  }));

  // PR curve (recall vs precision) for the best model on test.
  const pr = data.best_model.curves?.pr;
  const prData = pr ? pr.recall.map((r, i) => ({ recall: +r, precision: +pr.precision[i] })) : [];

  const tt = { contentStyle: { background: "#1e293b", border: "1px solid #334155" } };
  const cmCells = [
    ["True Negative", cm[0][0], "#1e293b"], ["False Positive", cm[0][1], "#7f1d1d"],
    ["False Negative", cm[1][0], "#7f1d1d"], ["True Positive", cm[1][1], "#14532d"],
  ];

  return (
    <div>
      <h1>Threshold Tuning</h1>
      <p className="subtitle">
        We do not use the default 0.5 cut-off. Thresholds 0.05–0.95 are evaluated on the
        validation set; the one maximising F1 for Declined is selected:{" "}
        <span className="pill">selected threshold = {selected}</span>
      </p>

      <div className="panel">
        <h2 style={{ marginTop: 0 }}>Precision / Recall / F1 vs Threshold (validation)</h2>
        <ResponsiveContainer width="100%" height={340}>
          <LineChart data={chartData}>
            <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
            <XAxis dataKey="threshold" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" domain={[0, 1]} />
            <Tooltip {...tt} />
            <Legend />
            <Line type="monotone" dataKey="Precision" stroke="#3b82f6" dot={false} />
            <Line type="monotone" dataKey="Recall" stroke="#ef4444" dot={false} />
            <Line type="monotone" dataKey="F1" stroke="#22c55e" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="Balanced Acc" stroke="#f59e0b" dot={false} />
            <ReferenceLine x={selected} stroke="#e2e8f0" strokeDasharray="4 4"
              label={{ value: `selected ${selected}`, fill: "#e2e8f0", position: "top" }} />
          </LineChart>
        </ResponsiveContainer>
        <p className="note">
          Lower threshold → higher Recall (catch more declines) but lower Precision (more false alarms).
          Move it to match the business cost of a missed decline vs a false alarm.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: 20 }}>
        <div className="panel">
          <h2 style={{ marginTop: 0 }}>Precision–Recall Curve (best model, test)</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={prData}>
              <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
              <XAxis dataKey="recall" stroke="#94a3b8" type="number" domain={[0, 1]}
                label={{ value: "Recall", position: "insideBottom", offset: -3, fill: "#94a3b8" }} />
              <YAxis stroke="#94a3b8" domain={[0, 1]} />
              <Tooltip {...tt} />
              <Line type="monotone" dataKey="precision" stroke="#8b5cf6" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="panel">
          <h2 style={{ marginTop: 0 }}>Confusion Matrix @ {selected} (test)</h2>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            {cmCells.map(([label, val, bg]) => (
              <div key={label} style={{ background: bg, borderRadius: 10, padding: 16, textAlign: "center" }}>
                <div className="note">{label}</div>
                <div style={{ fontSize: 30, fontWeight: 700 }}>{val}</div>
              </div>
            ))}
          </div>
          <p className="note" style={{ marginTop: 12 }}>
            Rows = actual, derived from the held-out test set at the selected threshold.
          </p>
        </div>
      </div>
    </div>
  );
}
