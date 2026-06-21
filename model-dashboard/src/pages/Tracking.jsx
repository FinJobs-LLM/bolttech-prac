import { useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";

export default function Tracking({ data }) {
  const models = Object.keys(data.optuna);
  const [model, setModel] = useState(data.best_model.model || models[0]);
  const info = data.optuna[model];

  // Running best (so the search "progress" is visible) + per-trial values.
  let best = -Infinity;
  const history = info.history
    .filter((h) => h.value != null)
    .map((h) => {
      best = Math.max(best, h.value);
      return { trial: h.trial, value: +h.value.toFixed(4), best: +best.toFixed(4) };
    });

  const importances = Object.entries(info.param_importances || {})
    .map(([param, importance]) => ({ param, importance: +importance.toFixed(4) }))
    .sort((a, b) => b.importance - a.importance);

  const tt = { contentStyle: { background: "#1e293b", border: "1px solid #334155" } };

  return (
    <div>
      <h1>MLflow / Optuna Tracking</h1>
      <p className="subtitle">
        Hyperparameter search history per model family. The objective maximised is
        validation PR-AUC. Full run details (params, metrics, artifacts) live in the MLflow UI.
      </p>

      <div style={{ marginBottom: 18 }}>
        <label className="field" style={{ maxWidth: 260 }}>
          <span className="name">Model family</span>
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            {models.map((m) => <option key={m}>{m}</option>)}
          </select>
        </label>
      </div>

      <div className="card-grid" style={{ marginBottom: 20 }}>
        <div className="card"><div className="label">Best validation PR-AUC</div>
          <div className="value">{info.best_value.toFixed(4)}</div></div>
        <div className="card"><div className="label">Trials</div>
          <div className="value">{info.n_trials}</div></div>
      </div>

      <div className="panel">
        <h2 style={{ marginTop: 0 }}>Search history — PR-AUC across trials</h2>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={history}>
            <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
            <XAxis dataKey="trial" stroke="#94a3b8" label={{ value: "Trial", position: "insideBottom", offset: -3, fill: "#94a3b8" }} />
            <YAxis stroke="#94a3b8" domain={["auto", "auto"]} />
            <Tooltip {...tt} />
            <Line type="monotone" dataKey="value" stroke="#64748b" dot={{ r: 2 }} name="trial PR-AUC" />
            <Line type="stepAfter" dataKey="best" stroke="#22c55e" strokeWidth={2} dot={false} name="running best" />
          </LineChart>
        </ResponsiveContainer>
        <p className="note">Grey: each trial's score. Green: best score found so far (monotonic).</p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
        <div className="panel">
          <h2 style={{ marginTop: 0 }}>Parameter importance</h2>
          {importances.length ? (
            <ResponsiveContainer width="100%" height={Math.max(220, importances.length * 34)}>
              <BarChart data={importances} layout="vertical" margin={{ left: 20 }}>
                <XAxis type="number" stroke="#94a3b8" />
                <YAxis type="category" dataKey="param" stroke="#94a3b8" width={130} />
                <Tooltip {...tt} />
                <Bar dataKey="importance" fill="#8b5cf6" />
              </BarChart>
            </ResponsiveContainer>
          ) : <p className="note">Not enough completed trials to compute importances.</p>}
          <p className="note">How much each hyperparameter influenced validation PR-AUC (fANOVA).</p>
        </div>

        <div className="panel">
          <h2 style={{ marginTop: 0 }}>Best trial — selected hyperparameters</h2>
          <pre>{JSON.stringify(info.best_params, null, 2)}</pre>
        </div>
      </div>
    </div>
  );
}
