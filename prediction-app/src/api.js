// Data layer for the standalone prediction app.
//
// - Feature metadata (defaults + categorical options + default threshold) is
//   bundled in public/feature_config.json so the form can render without the
//   backend.
// - Inference calls the FastAPI backend via the Vite dev proxy at /api
//   (configurable through VITE_API_BASE).

const API_BASE = import.meta.env.VITE_API_BASE || "/api";

// fetch() with a timeout so a stalled request surfaces an error instead of
// hanging the UI forever (e.g. a dead dev-server/tunnel connection).
async function fetchWithTimeout(url, opts = {}, ms = 30000) {
  const ctrl = new AbortController();
  const id = setTimeout(() => ctrl.abort(), ms);
  try {
    return await fetch(url, { ...opts, signal: ctrl.signal });
  } catch (e) {
    if (e.name === "AbortError") {
      throw new Error(
        `Request to ${url} timed out after ${ms / 1000}s. Is the FastAPI backend running, ` +
          `and (if using VS Code port forwarding) did you hard-refresh the page?`
      );
    }
    throw new Error(`Network error calling ${url}: ${e.message}`);
  } finally {
    clearTimeout(id);
  }
}

export async function loadConfig() {
  const res = await fetchWithTimeout("/feature_config.json", { cache: "no-store" });
  if (!res.ok) throw new Error("Could not load feature_config.json");
  return await res.json();
}

// Fetch the current best model's info from the FastAPI backend.
// Works against either backend:
//   - serve_api.py -> GET /metadata (richer: version, training date, ...)
//   - serve.py     -> GET /model-summary (fallback)
// Returns a normalized shape, or throws if neither endpoint is reachable.
export async function getModelInfo() {
  // Try the richer /metadata first.
  try {
    const res = await fetchWithTimeout(`${API_BASE}/metadata`);
    if (res.ok) {
      const d = await res.json();
      return {
        source: "metadata",
        model_name: d.model_name,
        stage: d.stage,
        model_version: d.model_version ?? null,
        training_date: d.training_date ?? null,
        threshold: d.decision_threshold,
        imbalance_strategy: d.imbalance_strategy,
        optuna_trials: d.optuna_trials ?? null,
        test_metrics: d.test_metrics ?? {},
      };
    }
  } catch (_) {
    /* fall through to /model-summary */
  }

  const res = await fetchWithTimeout(`${API_BASE}/model-summary`);
  if (!res.ok) throw new Error(`Could not load model info (${res.status})`);
  const d = await res.json();
  return {
    source: "model-summary",
    model_name: d.best_model,
    stage: d.stage,
    model_version: null,
    training_date: null,
    threshold: d.threshold,
    imbalance_strategy: d.imbalance_strategy,
    optuna_trials: null,
    test_metrics: d.test_metrics ?? {},
  };
}

// Fetch the serving model's feature importance.
//   - serve_api.py exposes it inside GET /metadata
//   - serve.py     exposes a dedicated GET /feature-importance
// Returns { model, items: [{feature, importance}] } (items sorted desc).
export async function getFeatureImportance() {
  try {
    const res = await fetchWithTimeout(`${API_BASE}/metadata`);
    if (res.ok) {
      const d = await res.json();
      if (Array.isArray(d.feature_importance) && d.feature_importance.length) {
        return { model: d.model_name, items: d.feature_importance };
      }
    }
  } catch (_) {
    /* fall through */
  }
  const res = await fetchWithTimeout(`${API_BASE}/feature-importance`);
  if (!res.ok) throw new Error(`Could not load feature importance (${res.status})`);
  const d = await res.json();
  return { model: d.model, items: d.feature_importance || [] };
}

// Fetch the LLM-generated explanation of the serving model (gpt-4o-mini via
// LangChain, server-side). Returns { explanation, cached, model, model_version }.
// Throws with a readable message (e.g. when the server has no OPENAI_API_KEY).
export async function getExplanation(refresh = false) {
  const res = await fetchWithTimeout(`${API_BASE}/explain${refresh ? "?refresh=true" : ""}`);
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const j = await res.json();
      msg = j.detail || j.error?.message || msg; // serve.py vs serve_api shapes
    } catch (_) {
      /* keep status */
    }
    throw new Error(msg);
  }
  return await res.json();
}

export async function predict(features, threshold) {
  const body = { features };
  if (threshold != null) body.threshold = threshold;
  const res = await fetchWithTimeout(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = await res.text();
    try {
      const j = JSON.parse(detail);
      detail = j.detail || j.error?.message || detail;
    } catch (_) {
      /* keep raw text */
    }
    throw new Error(`Prediction failed (${res.status}): ${detail}`);
  }
  return await res.json();
}

// Claims-adjuster explanation of THIS claim's prediction. The backend recomputes
// the model prediction and the LLM only explains it (it never decides).
// Returns { prediction, explanation }.
export async function explainPrediction(features, threshold) {
  const body = { features };
  if (threshold != null) body.threshold = threshold;
  const res = await fetchWithTimeout(`${API_BASE}/explain-prediction`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const j = await res.json();
      msg = j.detail || j.error?.message || msg;
    } catch (_) {
      /* keep status */
    }
    throw new Error(msg);
  }
  return await res.json();
}
