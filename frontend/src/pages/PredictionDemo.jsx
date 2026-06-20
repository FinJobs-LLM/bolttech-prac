import { useMemo, useState } from "react";
import { predict } from "../api/client.js";

export default function PredictionDemo({ data }) {
  const fm = data.overview.feature_meta || {};
  const features = Object.keys(fm);
  const initial = useMemo(() => {
    const o = {};
    for (const f of features) o[f] = fm[f].default;
    return o;
  }, [data]);

  const [values, setValues] = useState(initial);
  const [thr, setThr] = useState(data.best_model.threshold);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const setVal = (f, v) => setValues((s) => ({ ...s, [f]: v }));

  const run = async () => {
    setBusy(true); setError(null); setResult(null);
    // Coerce numeric fields back to numbers.
    const payload = {};
    for (const f of features) {
      const meta = fm[f];
      payload[f] = meta.type === "numeric" && values[f] !== "" ? Number(values[f]) : values[f];
    }
    try {
      setResult(await predict(payload, Number(thr)));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <h1>Prediction Demo</h1>
      <p className="subtitle">
        Enter a claim's feature values and the model returns the probability of Declined and the
        decision at the chosen threshold. Requires the FastAPI backend (see README).
      </p>

      <div className="panel">
        <div className="form-grid">
          {features.map((f) => {
            const meta = fm[f];
            return (
              <label className="field" key={f}>
                <span className="name">{f}</span>
                {meta.type === "categorical" ? (
                  <select value={values[f]} onChange={(e) => setVal(f, e.target.value)}>
                    {meta.options.map((o) => <option key={o} value={o}>{o}</option>)}
                  </select>
                ) : (
                  <input type="number" step="any" value={values[f]}
                    onChange={(e) => setVal(f, e.target.value)} />
                )}
              </label>
            );
          })}
        </div>

        <div style={{ display: "flex", gap: 16, alignItems: "flex-end", marginTop: 8 }}>
          <label className="field" style={{ maxWidth: 220, marginBottom: 0 }}>
            <span className="name">Decision threshold</span>
            <input type="number" step="0.01" min="0" max="1" value={thr}
              onChange={(e) => setThr(e.target.value)} />
          </label>
          <button className="primary" onClick={run} disabled={busy}>
            {busy ? "Scoring…" : "Predict"}
          </button>
        </div>
      </div>

      {error && <div className="warn-box">{error}</div>}

      {result && (
        <div className="panel">
          <h2 style={{ marginTop: 0 }}>
            Prediction:{" "}
            <span className="badge" style={{
              background: result.predicted_label ? "var(--bad)" : "var(--good)",
              color: result.predicted_label ? "white" : "#06281a",
            }}>
              {result.predicted_class}
            </span>
          </h2>
          <div className="result-box">
            <div className="card"><div className="label">P(Declined)</div>
              <div className="value">{(result.probability_declined * 100).toFixed(1)}%</div></div>
            <div className="card"><div className="label">P(Completed)</div>
              <div className="value">{(result.probability_completed * 100).toFixed(1)}%</div></div>
            <div className="card"><div className="label">Threshold used</div>
              <div className="value">{result.threshold_used}</div></div>
          </div>
          <p className="explain" style={{ marginTop: 14 }}>{result.explanation}</p>
        </div>
      )}
    </div>
  );
}
