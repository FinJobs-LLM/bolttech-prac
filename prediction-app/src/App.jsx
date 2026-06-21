import { useEffect, useMemo, useState } from "react";
import {
  explainPrediction, getExplanation, getFeatureImportance, getModelInfo, loadConfig, predict,
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
    setAdjExpl(null); // new prediction -> clear any prior adjuster explanation
    setAdjError(null);
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

  const resetDefaults = () => {
    const init = {};
    for (const [name, meta] of Object.entries(config.feature_meta)) init[name] = meta.default;
    setValues(init);
    setThr(modelInfo?.threshold ?? config.default_threshold);
    setResult(null);
    setError(null);
    setAdjExpl(null);
    setAdjError(null);
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

  return (
    <div className="page">
      <header>
        <h1>Claim Approval — Prediction</h1>
        <p className="subtitle">
          Enter a claim's feature values and get the model's prediction — positive class{" "}
          <strong>Declined</strong>.
        </p>
      </header>

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

          <div className="adjuster">
            <h3>
              Claims-adjuster explanation <span className="badge ai">gpt-4o-mini</span>
            </h3>
            <p className="note" style={{ marginTop: 0 }}>
              A professional explanation of this prediction for manual review. The model made the
              decision; this only explains it — it does not approve or decline the claim.
            </p>
            {!adjExpl && (
              <button className="primary" onClick={explainForAdjuster} disabled={adjLoading}>
                {adjLoading ? "Generating…" : "Explain for claims adjuster"}
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
          </div>
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
