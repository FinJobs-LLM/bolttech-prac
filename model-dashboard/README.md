# Model-Optimization Dashboard

One of **two independent front-end apps** in this repository (the other is [`../prediction-app/`](../prediction-app/)).

This app is the **analytics / explainability dashboard** for the claim-approval ML pipeline: it
visualizes how the models were trained, optimized and selected so a technical or semi-technical
audience can understand *why* the chosen model and threshold were picked. It is **not** the
operational claim-scoring app — that is `prediction-app`.

## What it shows (7 pages)
- **Overview** — dataset size, class distribution, imbalance ratio, train/val/test split.
- **Model Leaderboard** — all baseline & optimized models ranked by validation PR-AUC (Accuracy de-emphasized).
- **MLflow / Optuna** — per-family hyperparameter search history, running-best curve, parameter importance.
- **Metric Explanation** — business-friendly definitions (PR-AUC, Recall, Precision, F1, …).
- **Threshold Tuning** — precision/recall/F1 vs threshold, PR curve, confusion matrix at the chosen threshold.
- **Final Model** — selected model, hyperparameters, test metrics, feature importance, rationale.
- **Prediction Demo** — enter feature values and get a single prediction.

## How it relates to the rest of the repo
- Reads `public/dashboard_data.json`, which is produced by the Python pipeline
  (`src/run_pipeline.py`) and copied into this app's `public/` automatically at the end of a run.
- The **Prediction Demo** page calls the FastAPI backend (`src/serve.py`) at `/api/predict`.
- All other pages render from the static JSON and work **without** the backend running.

## Install
Node 18+ required.
```bash
cd model-dashboard
npm install
```

## Run locally (dev)
```bash
npm run dev          # http://localhost:5173
```
The Vite dev server proxies `/api/*` → `http://localhost:8000` (the `serve.py` backend), so the
Prediction Demo works without CORS issues. Start the backend separately if you want that page live:
`uv run uvicorn serve:app --app-dir src --port 8000` (from the repo root).

## Build
```bash
npm run build        # outputs to dist/
npm run preview      # serve the production build locally
```

## Configuration / environment
- `VITE_API_BASE` — base URL for backend calls (default `/api`). The dev proxy maps `/api` → `:8000`.
  For a production static build with the backend on another origin, set e.g.
  `VITE_API_BASE=https://your-api.example.com npm run build`, or put a reverse proxy in front that
  routes `/api/*` to the backend.

## Notes
- If `public/dashboard_data.json` is missing, run the pipeline first (`python src/run_pipeline.py`);
  the app shows a clear message until then.
- Stack: Vite + React + Recharts. `node_modules/` and `dist/` are git-ignored.
