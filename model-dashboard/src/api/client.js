// Data access layer.
//
// Dashboard data is loaded from the static bundle written by the training
// pipeline (model-dashboard/public/dashboard_data.json). The prediction demo calls the
// FastAPI backend through the Vite dev proxy at /api, falling back to a direct
// localhost:8000 call when not proxied.

const API_BASE = import.meta.env.VITE_API_BASE || "/api";

export async function loadDashboard() {
  // Prefer the static bundle (works without the backend running).
  try {
    const res = await fetch("/dashboard_data.json", { cache: "no-store" });
    if (res.ok) return await res.json();
  } catch (e) {
    /* fall through to API */
  }
  const res = await fetch(`${API_BASE}/dashboard`);
  if (!res.ok) throw new Error("Could not load dashboard data");
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
    const txt = await res.text();
    throw new Error(`Prediction failed (${res.status}): ${txt}`);
  }
  return await res.json();
}
