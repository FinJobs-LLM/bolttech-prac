import { useEffect, useMemo, useState } from "react";
import { loadConfig, predict } from "./api.js";

export default function App() {
  const [config, setConfig] = useState(null);
  const [configError, setConfigError] = useState(null);

  const [values, setValues] = useState({});
  const [thr, setThr] = useState(0.5);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  // Load bundled feature metadata once.
  useEffect(() => {
    loadConfig()
      .then((cfg) => {
        setConfig(cfg);
        const init = {};
        for (const [name, meta] of Object.entries(cfg.feature_meta)) init[name] = meta.default;
        setValues(init);
        setThr(cfg.default_threshold);
      })
      .catch((e) => setConfigError(e.message));
  }, []);

  const features = useMemo(
    () => (config ? Object.keys(config.feature_meta) : []),
    [config]
  );

  const setVal = (f, v) => setValues((s) => ({ ...s, [f]: v }));

  const run = async () => {
    setBusy(true);
    setError(null);
    setResult(null);
    const fm = config.feature_meta;
    const payload = {};
    for (const f of features) {
      const meta = fm[f];
      payload[f] =
        meta.type === "numeric" && values[f] !== "" ? Number(values[f]) : values[f];
    }
    try {
      setResult(await predict(payload, Number(thr)));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const resetDefaults = () => {
    const init = {};
    for (const [name, meta] of Object.entries(config.feature_meta)) init[name] = meta.default;
    setValues(init);
    setThr(config.default_threshold);
    setResult(null);
    setError(null);
  };

  if (configError) {
    return (
      <div className="page">
        <h1>Claim Approval — Prediction</h1>
        <div className="warn-box">
          {configError}. Make sure <code>public/feature_config.json</code> exists (generated from
          the trained model's metadata).
        </div>
      </div>
    );
  }

  if (!config) {
    return (
      <div className="page">
        <h1>Claim Approval — Prediction</h1>
        <div className="loading">Loading…</div>
      </div>
    );
  }

  return (
    <div className="page">
      <header>
        <h1>Claim Approval — Prediction</h1>
        <p className="subtitle">
          Enter a claim's feature values and get the model's prediction. Model:{" "}
          <span className="pill">
            {config.model} ({config.stage})
          </span>{" "}
          — positive class <strong>Declined</strong>.
        </p>
      </header>

      <section className="panel">
        <div className="form-grid">
          {features.map((f) => {
            const meta = config.feature_meta[f];
            return (
              <label className="field" key={f}>
                <span className="name">{f}</span>
                {meta.type === "categorical" ? (
                  <select value={values[f]} onChange={(e) => setVal(f, e.target.value)}>
                    {meta.options.map((o) => (
                      <option key={o} value={o}>
                        {o}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="number"
                    step="any"
                    value={values[f]}
                    onChange={(e) => setVal(f, e.target.value)}
                  />
                )}
              </label>
            );
          })}
        </div>

        <div className="controls">
          <label className="field" style={{ maxWidth: 220, marginBottom: 0 }}>
            <span className="name">Decision threshold</span>
            <input
              type="number"
              step="0.01"
              min="0"
              max="1"
              value={thr}
              onChange={(e) => setThr(e.target.value)}
            />
          </label>
          <button className="primary" onClick={run} disabled={busy}>
            {busy ? "Scoring…" : "Predict"}
          </button>
          <button className="ghost" onClick={resetDefaults} disabled={busy}>
            Reset
          </button>
        </div>
      </section>

      {error && <div className="warn-box">{error}</div>}

      {result && (
        <section className="panel">
          <h2>
            Prediction:{" "}
            <span
              className="badge"
              style={{
                background: result.predicted_label ? "#ef4444" : "#22c55e",
                color: result.predicted_label ? "white" : "#06281a",
              }}
            >
              {result.predicted_class}
            </span>
          </h2>
          <div className="result-box">
            <div className="card">
              <div className="label">P(Declined)</div>
              <div className="value">{(result.probability_declined * 100).toFixed(1)}%</div>
            </div>
            <div className="card">
              <div className="label">P(Completed)</div>
              <div className="value">{(result.probability_completed * 100).toFixed(1)}%</div>
            </div>
            <div className="card">
              <div className="label">Threshold used</div>
              <div className="value">{result.threshold_used}</div>
            </div>
          </div>
          {result.explanation && <p className="explain">{result.explanation}</p>}
        </section>
      )}
    </div>
  );
}
