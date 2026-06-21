# Claim Approval — Standalone Prediction App

A small, **independent** front-end that does one thing: let a user enter a claim's feature values and
see the model's prediction (predicted class, P(Declined), P(Completed), threshold used).

This is a separate project from `../frontend/` (the full dashboard) and does not depend on it. It
replicates only the "Prediction Demo" functionality.

## How it works
- Feature inputs (defaults + categorical options + default threshold) are bundled in
  `public/feature_config.json`, generated from the trained model's metadata — so the form renders
  without any backend.
- Inference calls the FastAPI backend via the Vite dev proxy at `/api/predict`.

## Run

```bash
cd prediction-app
npm install
npm run dev            # http://localhost:5174
```

The proxy targets `http://localhost:8000` (the `serve.py` backend) by default. To target the
production serving API (`serve_api.py`, port 8001) instead:

```bash
API_TARGET=http://localhost:8001 npm run dev
```

Make sure one of the backends is running (see the main project README), e.g.:

```bash
uv run uvicorn serve:app --app-dir src --port 8000
```

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
