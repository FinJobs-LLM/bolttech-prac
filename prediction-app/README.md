# Claim Approval — Standalone Prediction App

A focused, **independent** front-end for **claim prediction and review**: score a claim, then review
the result with adjuster/customer explanations, record an adjuster decision, and browse history — all
backed by the FastAPI service.

This is a separate project from `../model-dashboard/` (the analytics dashboard) and does not depend on
it. The model always makes the prediction; the LLM only *explains* it.

## Tabs
The app has a tab menu (in this order):
- **Prediction** — enter claim features, pick a threshold, get the model's prediction. When the
  database is configured, each prediction is saved (response includes `prediction_id`, `saved_to_db`).
- **Model & Features** — live model card and feature-importance bar chart, plus an AI explanation of
  the model (`GET /explain`).
- **Claims Adjuster** — an adjuster-focused LLM explanation of the *current* prediction
  (`POST /explain-prediction`), a **Save explanation to database** button
  (`POST /predictions/{id}/explanation`), and **Approve → Completed / Decline → Declined** buttons that
  record the adjuster's final decision (`POST /predictions/{id}/decision`). The model's prediction is
  preserved; the decision is stored separately.
- **Customer** — a plain-language, customer-friendly explanation of the *current* prediction
  (`POST /explain-prediction-customer`): summarises the preliminary, automated result and the main
  claim details that influenced it, in simple words with no technical/insurance jargon. The model
  decides; the LLM only explains, never finalises the decision, and gives no legal/financial/coverage
  advice.
- **History** — recent predictions saved to the database (`GET /predictions/recent`): all input
  features, the model result, the adjuster decision, and any saved explanations. Shows a notice if DB
  saving isn't configured on the server.

## How it works
- Feature inputs (defaults + categorical options) are bundled in `public/feature_config.json`,
  generated from the trained model's metadata — so the form renders without any backend.
- On load it fetches the **current best model's info live from the API** and shows a model card
  (name/stage, version + training date when available, imbalance strategy, decision threshold, and
  held-out test metrics). It tries `GET /metadata` (prediction_service_api.py — richest) and falls back to
  `GET /model-summary` (dashboard_api.py). The live decision threshold is used as the form default.
- It also shows the serving model's **feature importance** as a bar chart (rendered with plain CSS —
  no chart library), fetched live from `GET /metadata` (prediction_service_api.py) or `GET /feature-importance`
  (dashboard_api.py).
- An **AI explanation** panel (button-triggered) calls `GET /explain`, which generates a plain-English
  summary of the model + its feature importance server-side using LangChain + `gpt-4o-mini`. This
  requires the backend to have `OPENAI_API_KEY` set; without it the panel shows a clear notice and the
  rest of the app still works.
- After a prediction, a **Claims-adjuster explanation** (button) calls `POST /explain-prediction`,
  which generates an adjuster-facing explanation of *this* prediction (LangChain + `gpt-4o-mini`),
  combining the prediction result, model reliability metrics, feature importance, and the claim's
  values. The model makes the decision; the LLM only explains it for manual review and never
  approves/declines the claim. The backend recomputes the prediction so the explanation always
  matches the real model output.
- Inference calls the FastAPI backend via the Vite dev proxy at `/api/predict`.

If the API is unreachable, the model card shows a notice and the form still works using the bundled
defaults.

## Run

```bash
cd prediction-app
npm install
npm run dev            # http://localhost:5174
```

The proxy targets `http://localhost:8000` (the `dashboard_api.py` backend) by default. To target the
production serving API (`prediction_service_api.py`, port 8001) instead:

```bash
API_TARGET=http://localhost:8001 npm run dev
```

Make sure one of the backends is running (see the main project README), e.g.:

```bash
uv run uvicorn dashboard_api:app --app-dir src --port 8000
```

## Production build (important)

The `/api` proxy above is a **Vite dev-server feature** — it only exists during `npm run dev`. A
production `npm run build` produces static files with no proxy, so you must tell the app where the
FastAPI backend lives in one of two ways:

1. **Bake an absolute API URL at build time** via `VITE_API_BASE`:
   ```bash
   VITE_API_BASE=https://your-api.example.com npm run build
   ```
   (`api.js` reads `import.meta.env.VITE_API_BASE`, falling back to `/api` when unset.)

2. **Keep the relative `/api` path and add a reverse proxy** (nginx, Caddy, a CDN rule, etc.) in
   front of the static files that forwards `/api/*` to the FastAPI server.

Inference always runs in FastAPI (`/predict`) — the app never scores locally — so a reachable
backend is required in every environment.

## Regenerating the feature config
If the model is retrained, refresh the bundled config from `reports/dashboard_data.json`:

```python
import json
d = json.load(open("../reports/dashboard_data.json"))
json.dump({
    "model": d["best_model"]["model"],
    "stage": d["best_model"]["stage"],
    "default_threshold": d["best_model"]["threshold"],
    "feature_meta": d["overview"]["feature_meta"],
}, open("public/feature_config.json", "w"), indent=2)
```
