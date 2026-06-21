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
