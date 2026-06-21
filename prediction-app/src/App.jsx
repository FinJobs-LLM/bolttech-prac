import { useEffect, useMemo, useState } from "react";
import {
  explainPrediction, explainPredictionCustomer, getExplanation, getFeatureImportance,
  getModelInfo, getRecentPredictions, loadConfig, predict,
} from "./api.js";

export default function App() {
  const [config, setConfig] = useState(null);
  const [configError, setConfigError] = useState(null);

  const [modelInfo, setModelInfo] = useState(null);
  const [modelInfoError, setModelInfoError] = useState(null);

  const [importance, setImportance] = useState(null);
  const [importanceError, setImportanceError] = useState(null);

  const [explanation, setExplanation] = useState(null);
  const [explLoading, setExplLoading] = useState(false);
  const [explError, setExplError] = useState(null);

  const [values, setValues] = useState({});
  const [thr, setThr] = useState(0.5);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  // Adjuster-facing explanation of the current prediction.
  const [adjExpl, setAdjExpl] = useState(null);
  const [adjLoading, setAdjLoading] = useState(false);
  const [adjError, setAdjError] = useState(null);

  // Customer-facing explanation of the current prediction.
  const [custExpl, setCustExpl] = useState(null);
  const [custLoading, setCustLoading] = useState(false);
  const [custError, setCustError] = useState(null);

  // Saved-prediction history (from the database).
  const [history, setHistory] = useState(null); // {enabled, count, predictions}
  const [histLoading, setHistLoading] = useState(false);
  const [histError, setHistError] = useState(null);

  const [tab, setTab] = useState("predict"); // predict | model | adjuster | customer | history

  // Load bundled feature metadata once (powers the input form).
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

  // Fetch the current best model's info live from the API (for display + the
  // live decision threshold). Non-fatal: the form still works if it fails.
  useEffect(() => {
    getModelInfo()
      .then((info) => {
        setModelInfo(info);
        if (info.threshold != null) setThr(info.threshold);
      })
      .catch((e) => setModelInfoError(e.message));
  }, []);

  // Fetch the serving model's feature importance (for display). Non-fatal.
  useEffect(() => {
    getFeatureImportance()
      .then(setImportance)
      .catch((e) => setImportanceError(e.message));
  }, []);

  // Refresh saved-prediction history whenever the History tab is opened.
  useEffect(() => {
    if (tab === "history") loadHistory();
  }, [tab]);

  const features = useMemo(
    () => (config ? Object.keys(config.feature_meta) : []),
    [config]
  );

  const setVal = (f, v) => setValues((s) => ({ ...s, [f]: v }));

  // Coerce form values into the payload sent to the API (numbers stay numbers).
  const buildPayload = () => {
    const fm = config.feature_meta;
    const payload = {};
    for (const f of features) {
      const meta = fm[f];
      payload[f] =
        meta.type === "numeric" && values[f] !== "" ? Number(values[f]) : values[f];
    }
    return payload;
  };

  const run = async () => {
    setBusy(true);
    setError(null);
    setResult(null);
    setAdjExpl(null); // new prediction -> clear any prior explanations
    setAdjError(null);
    setCustExpl(null);
    setCustError(null);
    try {
      setResult(await predict(buildPayload(), Number(thr)));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const explainForAdjuster = async () => {
    setAdjLoading(true);
    setAdjError(null);
    try {
      const d = await explainPrediction(buildPayload(), Number(thr));
      setAdjExpl(d.explanation);
    } catch (e) {
      setAdjError(e.message);
    } finally {
      setAdjLoading(false);
    }
  };

  const loadHistory = async () => {
    setHistLoading(true);
    setHistError(null);
    try {
      setHistory(await getRecentPredictions(25));
    } catch (e) {
      setHistError(e.message);
    } finally {
      setHistLoading(false);
    }
  };

  const explainForCustomer = async () => {
    setCustLoading(true);
    setCustError(null);
    try {
      const d = await explainPredictionCustomer(buildPayload(), Number(thr));
      setCustExpl(d.explanation);
    } catch (e) {
      setCustError(e.message);
    } finally {
      setCustLoading(false);
    }
  };

  const resetDefaults = () => {
    const init = {};
    for (const [name, meta] of Object.entries(config.feature_meta)) init[name] = meta.default;
    setValues(init);
    setThr(modelInfo?.threshold ?? config.default_threshold);
    setResult(null);
    setError(null);
    setAdjExpl(null);
    setAdjError(null);
    setCustExpl(null);
    setCustError(null);
  };

  const generateExplanation = async (refresh = false) => {
    setExplLoading(true);
    setExplError(null);
    try {
      // refresh=true forces the backend to call gpt-4o-mini again (bypass cache).
      const d = await getExplanation(refresh);
      setExplanation(d.explanation);
    } catch (e) {
      setExplError(e.message);
    } finally {
      setExplLoading(false);
    }
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

  const TABS = [
    ["predict", "Prediction"],
    ["model", "Model & Features"],
    ["adjuster", "Claims Adjuster"],
    ["customer", "Customer"],
    ["history", "History"],
  ];

  return (
    <div className="page">
      <header>
        <h1>Claim Approval — Assistant</h1>
        <p className="subtitle">
          Predict a claim's <strong>status</strong> (positive class <strong>Declined</strong>),
          inspect the model, and get a claims-adjuster explanation of the result.
        </p>
      </header>

      <nav className="tabs">
        {TABS.map(([key, label]) => (
          <button
            key={key}
            className={"tab" + (tab === key ? " active" : "")}
            onClick={() => setTab(key)}
          >
            {label}
            {key === "adjuster" && adjExpl && <span className="dot" />}
            {key === "customer" && custExpl && <span className="dot" />}
          </button>
        ))}
      </nav>

      {/* ---------------- Prediction tab ---------------- */}
      {tab === "predict" && (
        <>
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
              <p className="note">
                For a manual-review explanation of this result, open the{" "}
                <button className="link" onClick={() => setTab("adjuster")}>
                  Claims Adjuster
                </button>{" "}
                tab.
              </p>
            </section>
          )}
        </>
      )}

      {/* ---------------- Model & Features tab ---------------- */}
      {tab === "model" && (
        <>
          <ModelInfo info={modelInfo} error={modelInfoError} fallback={config} />
          <FeatureImportance data={importance} error={importanceError} />
          <section className="panel">
            <h2>
              AI explanation <span className="badge ai">gpt-4o-mini</span>
            </h2>
            <p className="note" style={{ marginTop: 0 }}>
              A plain-English summary of the model and what drives it, generated on the server with
              LangChain. Click to generate (cached after the first request).
            </p>
            {!explanation && (
              <button className="primary" onClick={() => generateExplanation(false)} disabled={explLoading}>
                {explLoading ? "Generating…" : "Generate AI explanation"}
              </button>
            )}
            {explError && <div className="warn-box">{explError}</div>}
            {explanation && (
              <>
                <Explanation text={explanation} />
                <button className="ghost" onClick={() => generateExplanation(true)} disabled={explLoading}>
                  {explLoading ? "Regenerating…" : "Regenerate"}
                </button>
              </>
            )}
          </section>
        </>
      )}

      {/* ---------------- Claims Adjuster tab ---------------- */}
      {tab === "adjuster" && (
        <section className="panel">
          <h2>
            Claims-Adjuster Explanation <span className="badge ai">gpt-4o-mini</span>
          </h2>
          <p className="note" style={{ marginTop: 0 }}>
            A professional explanation of the current prediction for manual review. The ML model
            makes the decision; this only explains the model's output — it does not approve or
            decline the claim.
          </p>

          {!result ? (
            <div className="warn-box">
              No prediction yet. Enter a claim and click <strong>Predict</strong> on the{" "}
              <button className="link" onClick={() => setTab("predict")}>
                Prediction
              </button>{" "}
              tab, then return here for an adjuster-focused explanation of that result.
            </div>
          ) : (
            <>
              <div className="result-box">
                <div className="card">
                  <div className="label">Model output</div>
                  <div className="value small">{result.predicted_class}</div>
                </div>
                <div className="card">
                  <div className="label">P(Declined)</div>
                  <div className="value">{(result.probability_declined * 100).toFixed(1)}%</div>
                </div>
                <div className="card">
                  <div className="label">Threshold</div>
                  <div className="value">{result.threshold_used}</div>
                </div>
              </div>

              {!adjExpl && (
                <button className="primary" onClick={explainForAdjuster} disabled={adjLoading}>
                  {adjLoading ? "Generating…" : "Explain this result for a claims adjuster"}
                </button>
              )}
              {adjError && <div className="warn-box">{adjError}</div>}
              {adjExpl && (
                <>
                  <Explanation text={adjExpl} />
                  <button className="ghost" onClick={explainForAdjuster} disabled={adjLoading}>
                    {adjLoading ? "Regenerating…" : "Regenerate"}
                  </button>
                </>
              )}
            </>
          )}
        </section>
      )}

      {/* ---------------- Customer tab ---------------- */}
      {tab === "customer" && (
        <section className="panel">
          <h2>
            Explanation for the Customer <span className="badge ai">gpt-4o-mini</span>
          </h2>
          <p className="note" style={{ marginTop: 0 }}>
            A clear, plain-language message about the current claim's automated review, written for
            the customer. This is a preliminary, automated result that a person may review — it is
            not a final decision, and it is not legal, financial, or coverage advice.
          </p>

          {!result ? (
            <div className="warn-box">
              No prediction yet. Enter a claim and click <strong>Predict</strong> on the{" "}
              <button className="link" onClick={() => setTab("predict")}>
                Prediction
              </button>{" "}
              tab, then return here for a plain-language explanation for the customer.
            </div>
          ) : (
            <>
              <div className="result-box">
                <div className="card">
                  <div className="label">Automated review (preliminary)</div>
                  <div className="value small">
                    {result.predicted_label
                      ? "May not be approved"
                      : "Likely to be approved"}
                  </div>
                </div>
              </div>

              {!custExpl && (
                <button className="primary" onClick={explainForCustomer} disabled={custLoading}>
                  {custLoading ? "Writing…" : "Explain this result for the customer"}
                </button>
              )}
              {custError && <div className="warn-box">{custError}</div>}
              {custExpl && (
                <>
                  <Explanation text={custExpl} />
                  <button className="ghost" onClick={explainForCustomer} disabled={custLoading}>
                    {custLoading ? "Rewriting…" : "Regenerate"}
                  </button>
                </>
              )}
            </>
          )}
        </section>
      )}

      {/* ---------------- History tab ---------------- */}
      {tab === "history" && (
        <section className="panel">
          <h2 style={{ display: "flex", alignItems: "center", gap: 12 }}>
            Saved Predictions
            <button className="ghost" onClick={loadHistory} disabled={histLoading}
              style={{ padding: "6px 12px", fontSize: 13 }}>
              {histLoading ? "Loading…" : "Refresh"}
            </button>
          </h2>
          <p className="note" style={{ marginTop: 0 }}>
            The most recent predictions saved to the database (newest first). Every time you click
            Predict, the claim's features and the result are stored.
          </p>

          {histError && <div className="warn-box">{histError}</div>}
          {history && !history.enabled && (
            <div className="warn-box">
              Saving to the database is not configured on the server (no DB connection set).
            </div>
          )}
          {history && history.enabled && history.predictions.length === 0 && !histLoading && (
            <p className="note">No saved predictions yet — run a prediction first.</p>
          )}
          {history && history.enabled && history.predictions.length > 0 && (
            <div style={{ overflowX: "auto" }}>
              <table className="history">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Saved at</th>
                    <th>Result</th>
                    <th>P(Declined)</th>
                    <th>Threshold</th>
                    {features.map((f) => (
                      <th key={f}>{f}</th>
                    ))}
                    <th>Model</th>
                  </tr>
                </thead>
                <tbody>
                  {history.predictions.map((r) => (
                    <tr key={r.id}>
                      <td>{r.id}</td>
                      <td>{r.created_at ? new Date(r.created_at).toLocaleString() : "—"}</td>
                      <td>
                        <span className="badge" style={{
                          background: r.predicted_label ? "#ef4444" : "#22c55e",
                          color: r.predicted_label ? "white" : "#06281a",
                        }}>
                          {r.predicted_class}
                        </span>
                      </td>
                      <td>{r.probability_declined != null
                        ? (r.probability_declined * 100).toFixed(1) + "%" : "—"}</td>
                      <td>{r.threshold_used}</td>
                      {features.map((f) => (
                        <td key={f}>{r[f] ?? "—"}</td>
                      ))}
                      <td className="note">{r.model_version ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}
    </div>
  );
}

const pct = (v) => (v == null ? "—" : (v * 100).toFixed(1) + "%");
const num = (v) => (v == null ? "—" : Number(v).toFixed(3));

// Inline **bold** -> <strong>.
function inline(text, keyPrefix) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith("**") && part.endsWith("**") ? (
      <strong key={`${keyPrefix}-${i}`}>{part.slice(2, -2)}</strong>
    ) : (
      part
    )
  );
}

// Minimal Markdown renderer for the AI explanation (headings, bullet/numbered
// lists, bold). Dependency-free — the explanation is short and well-structured.
function Explanation({ text }) {
  const lines = text.split("\n");
  const blocks = [];
  let list = null;
  const flushList = () => {
    if (list) {
      blocks.push({ type: "ul", items: list });
      list = null;
    }
  };
  lines.forEach((raw) => {
    const line = raw.trim();
    if (!line) return flushList();
    const heading = line.match(/^#{1,6}\s+(.*)$/);
    const bullet = line.match(/^(?:[-*]|\d+\.)\s+(.*)$/);
    if (heading) {
      flushList();
      blocks.push({ type: "h", text: heading[1] });
    } else if (bullet) {
      (list ||= []).push(bullet[1]);
    } else {
      flushList();
      blocks.push({ type: "p", text: line });
    }
  });
  flushList();

  return (
    <div className="explanation">
      {blocks.map((b, i) => {
        if (b.type === "h") return <h4 key={i}>{inline(b.text, i)}</h4>;
        if (b.type === "ul")
          return (
            <ul key={i}>
              {b.items.map((it, j) => (
                <li key={j}>{inline(it, `${i}-${j}`)}</li>
              ))}
            </ul>
          );
        return <p key={i}>{inline(b.text, i)}</p>;
      })}
    </div>
  );
}

// Live "model card": the current best model's info fetched from the API.
function ModelInfo({ info, error, fallback }) {
  if (error && !info) {
    return (
      <section className="panel model-info">
        <h2>Current best model</h2>
        <p className="note">
          Live model info unavailable ({error}). Showing bundled defaults:{" "}
          <span className="pill">
            {fallback?.model} ({fallback?.stage})
          </span>
          . Start a FastAPI backend to load live info.
        </p>
      </section>
    );
  }
  if (!info) {
    return (
      <section className="panel model-info">
        <h2>Current best model</h2>
        <p className="note">Loading model info from the API…</p>
      </section>
    );
  }

  const m = info.test_metrics || {};
  const metrics = [
    ["PR-AUC", num(m.pr_auc)],
    ["ROC-AUC", num(m.roc_auc)],
    ["F1 (Declined)", num(m.f1_declined)],
    ["Recall (Declined)", pct(m.recall_declined)],
    ["Precision (Declined)", pct(m.precision_declined)],
    ["Balanced Acc", pct(m.balanced_accuracy)],
  ];

  return (
    <section className="panel model-info">
      <h2>
        Current best model{" "}
        <span className="badge live">live · /{info.source}</span>
      </h2>
      <div className="kv">
        <span className="k">Model</span>
        <span>
          {info.model_name} <span className="note">({info.stage})</span>
        </span>
        {info.model_version && (
          <>
            <span className="k">Version</span>
            <span>{info.model_version}</span>
          </>
        )}
        {info.training_date && (
          <>
            <span className="k">Trained</span>
            <span>{new Date(info.training_date).toLocaleString()}</span>
          </>
        )}
        <span className="k">Imbalance</span>
        <span>{info.imbalance_strategy}</span>
        <span className="k">Decision threshold</span>
        <span>{info.threshold}</span>
        {info.optuna_trials != null && (
          <>
            <span className="k">Optuna trials</span>
            <span>{info.optuna_trials}</span>
          </>
        )}
      </div>

      <div className="metric-row">
        {metrics.map(([label, val]) => (
          <div className="metric" key={label}>
            <div className="label">{label}</div>
            <div className="mval">{val}</div>
          </div>
        ))}
        <div className="metric de-emph">
          <div className="label">Accuracy</div>
          <div className="mval">{pct(m.accuracy)}</div>
        </div>
      </div>
      <p className="note">Test-set metrics (held out). Positive class = Declined.</p>
    </section>
  );
}

const TOP_N = 15;

// Live feature-importance bar chart (CSS bars — no chart library needed).
function FeatureImportance({ data, error }) {
  if (error && !data) {
    return (
      <section className="panel">
        <h2>Feature importance</h2>
        <p className="note">Live feature importance unavailable ({error}).</p>
      </section>
    );
  }
  if (!data) {
    return (
      <section className="panel">
        <h2>Feature importance</h2>
        <p className="note">Loading feature importance from the API…</p>
      </section>
    );
  }

  const items = [...(data.items || [])]
    .sort((a, b) => b.importance - a.importance)
    .slice(0, TOP_N);
  const max = items.length ? items[0].importance : 1;

  return (
    <section className="panel">
      <h2>
        Feature importance <span className="badge live">live</span>
      </h2>
      <p className="note" style={{ marginTop: 0 }}>
        How much the serving model ({data.model}) relies on each feature
        {data.items.length > TOP_N ? ` — top ${TOP_N} of ${data.items.length}` : ""}.
      </p>
      <div className="fi-list">
        {items.map((it) => (
          <div className="fi-row" key={it.feature}>
            <div className="fi-name" title={it.feature}>{it.feature}</div>
            <div className="fi-track">
              <div
                className="fi-bar"
                style={{ width: `${Math.max(2, (it.importance / max) * 100)}%` }}
              />
            </div>
            <div className="fi-score">{Number(it.importance).toFixed(2)}</div>
          </div>
        ))}
      </div>
    </section>
  );
}
