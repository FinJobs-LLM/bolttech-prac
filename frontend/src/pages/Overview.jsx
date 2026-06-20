import {
  Bar, BarChart, Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import Card from "../components/Card.jsx";

const COLORS = { Completed: "#22c55e", Declined: "#ef4444" };

export default function Overview({ data }) {
  const o = data.overview;
  const dist = Object.entries(o.target_distribution).map(([name, value]) => ({ name, value }));
  const splits = Object.entries(o.split_sizes).map(([name, value]) => ({ name, value }));

  return (
    <div>
      <h1>Dataset Overview</h1>
      <p className="subtitle">
        Predicting <strong>{`status`}</strong> — Declined (positive) vs Completed (negative).
      </p>

      <div className="card-grid">
        <Card label="Dataset" value={o.dataset_name} small />
        <Card label="Rows" value={o.n_rows.toLocaleString()} />
        <Card label="Features" value={o.n_features} hint={`${o.n_numeric} numeric · ${o.n_categorical} categorical`} />
        <Card label="Imbalance ratio" value={`${o.imbalance_ratio}:1`} hint={`weight ≈ ${o.imbalance_weight}`} />
        <Card label="Declined share"
          value={`${((o.class_counts.Declined / o.n_rows) * 100).toFixed(1)}%`}
          hint="minority / positive class" />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginTop: 24 }}>
        <div className="panel">
          <h2 style={{ marginTop: 0 }}>Class Distribution</h2>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={dist}>
              <XAxis dataKey="name" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155" }} />
              <Bar dataKey="value">
                {dist.map((d) => <Cell key={d.name} fill={COLORS[d.name] || "#3b82f6"} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="panel">
          <h2 style={{ marginTop: 0 }}>Completed vs Declined</h2>
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie data={dist} dataKey="value" nameKey="name" innerRadius={55} outerRadius={95} label>
                {dist.map((d) => <Cell key={d.name} fill={COLORS[d.name] || "#3b82f6"} />)}
              </Pie>
              <Legend />
              <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155" }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="panel">
        <h2 style={{ marginTop: 0 }}>Train / Validation / Test Split (stratified 70 / 15 / 15)</h2>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={splits} layout="vertical">
            <XAxis type="number" stroke="#94a3b8" />
            <YAxis type="category" dataKey="name" stroke="#94a3b8" width={80} />
            <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155" }} />
            <Bar dataKey="value" fill="#3b82f6" />
          </BarChart>
        </ResponsiveContainer>
        <p className="note">
          Validation is used for hyperparameter search, threshold tuning and early stopping.
          The test set is touched only once, for the final unbiased evaluation.
        </p>
      </div>
    </div>
  );
}
