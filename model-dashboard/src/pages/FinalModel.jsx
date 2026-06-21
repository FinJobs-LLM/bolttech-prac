import {
  Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import Card from "../components/Card.jsx";

const pct = (v) => (v == null ? "—" : (v * 100).toFixed(1) + "%");

export default function FinalModel({ data }) {
  const b = data.best_model;
  const tm = b.test_metrics;
  const imp = (b.feature_importance || []).slice(0, 15)
    .map((d) => ({ feature: d.feature, importance: +d.importance.toFixed(4) }));
  const rec = data.final_recommendation;

  // Render the explanation (markdown-ish) as paragraphs/bullets.
  const lines = (b.explanation || "").split("\n").filter((l) => l.trim());

  return (
    <div>
      <h1>Final Model</h1>
      <p className="subtitle">
        Selected by highest validation PR-AUC, then evaluated once on the held-out test set.
      </p>

      <div className="card-grid">
        <Card label="Best model" value={b.model} small hint={b.stage} />
        <Card label="Imbalance strategy" value={b.imbalance_strategy} small />
        <Card label="Threshold" value={b.threshold} />
        <Card label="Test PR-AUC" value={tm.pr_auc.toFixed(3)} />
        <Card label="Recall (Declined)" value={pct(tm.recall_declined)} hint="declines caught" />
        <Card label="Precision (Declined)" value={pct(tm.precision_declined)} />
        <Card label="Balanced Acc" value={pct(tm.balanced_accuracy)} />
        <Card label="Accuracy" value={pct(tm.accuracy)} hint="de-emphasised" />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginTop: 24 }}>
        <div className="panel">
          <h2 style={{ marginTop: 0 }}>Feature Importance (top 15)</h2>
          <ResponsiveContainer width="100%" height={Math.max(260, imp.length * 26)}>
            <BarChart data={imp} layout="vertical" margin={{ left: 30 }}>
              <XAxis type="number" stroke="#94a3b8" />
              <YAxis type="category" dataKey="feature" stroke="#94a3b8" width={150}
                tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155" }} />
              <Bar dataKey="importance" fill="#8b5cf6" />
            </BarChart>
          </ResponsiveContainer>
          <p className="note">One-hot columns are rolled back up to their original feature.</p>
        </div>

        <div className="panel">
          <h2 style={{ marginTop: 0 }}>Selected Hyperparameters</h2>
          <pre>{JSON.stringify(b.params, null, 2)}</pre>
        </div>
      </div>

      <div className="panel">
        <h2 style={{ marginTop: 0 }}>Why this model?</h2>
        <div className="explain">
          {lines.map((l, i) => {
            const txt = l.replace(/\*\*(.+?)\*\*/g, "‹$1›"); // keep it simple
            const html = l.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
            if (l.startsWith("- "))
              return <li key={i} dangerouslySetInnerHTML={{ __html: html.slice(2) }} />;
            return <p key={i} dangerouslySetInnerHTML={{ __html: html }} />;
          })}
        </div>
      </div>

      <div className="panel">
        <h2 style={{ marginTop: 0 }}>Final Recommendation</h2>
        <pre>{`Best Model: ${rec.best_model}
Best Imbalance Strategy: ${rec.imbalance_strategy}
Best Threshold: ${rec.threshold}
Main Reason: ${rec.reason}
Key Risk / Limitation: ${rec.risk}`}</pre>
      </div>
    </div>
  );
}
