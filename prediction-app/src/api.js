// Data layer for the standalone prediction app.
//
// - Feature metadata (defaults + categorical options + default threshold) is
//   bundled in public/feature_config.json so the form can render without the
//   backend.
// - Inference calls the FastAPI backend via the Vite dev proxy at /api
//   (configurable through VITE_API_BASE).

const API_BASE = import.meta.env.VITE_API_BASE || "/api";

export async function loadConfig() {
  const res = await fetch("/feature_config.json", { cache: "no-store" });
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
    const res = await fetch(`${API_BASE}/metadata`);
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

  const res = await fetch(`${API_BASE}/model-summary`);
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

export async function predict(features, threshold) {
  const body = { features };
  if (threshold != null) body.threshold = threshold;
  const res = await fetch(`${API_BASE}/predict`, {
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
