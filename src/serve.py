"""FastAPI service exposing the trained model and the dashboard data.

Run from the project root:
    .venv/bin/uvicorn serve:app --app-dir src --reload --port 8000

Endpoints:
    GET  /                    health / index
    POST /predict             score a single raw feature row
    GET  /model-summary       best model name, params, threshold, test metrics
    GET  /model-comparison    leaderboard of all trained models
    GET  /threshold-analysis  per-threshold precision/recall/F1/bal-acc (validation)
    GET  /feature-importance  feature importances for the selected model
    GET  /dashboard           the full dashboard_data.json bundle
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make sibling modules importable whether launched via uvicorn or directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import BEST_MODEL_META, BEST_MODEL_PATH, DASHBOARD_JSON
from model_factory import ClaimModel  # noqa: F401  (needed for joblib unpickling)

app = FastAPI(title="Claim Approval Model API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_MODEL: ClaimModel | None = None
_META: dict = {}
_DASH: dict = {}


def _load():
    global _MODEL, _META, _DASH
    if _MODEL is None and BEST_MODEL_PATH.exists():
        _MODEL = joblib.load(BEST_MODEL_PATH)
    if not _META and BEST_MODEL_META.exists():
        _META = json.loads(BEST_MODEL_META.read_text())
    if not _DASH and DASHBOARD_JSON.exists():
        _DASH = json.loads(DASHBOARD_JSON.read_text())


@app.on_event("startup")
def _startup():
    _load()


class PredictRequest(BaseModel):
    features: dict = Field(..., description="Raw feature name -> value mapping")
    threshold: float | None = Field(None, description="Override decision threshold")


@app.get("/")
def index():
    _load()
    return {
        "service": "Claim Approval Model API",
        "model_loaded": _MODEL is not None,
        "best_model": _META.get("model_type"),
        "endpoints": ["/predict", "/model-summary", "/model-comparison",
                      "/threshold-analysis", "/feature-importance", "/dashboard"],
    }


@app.post("/predict")
def predict(req: PredictRequest):
    _load()
    if _MODEL is None:
        raise HTTPException(503, "Model not trained yet. Run src/run_pipeline.py first.")
    # Build a one-row frame with all expected feature columns (missing -> None).
    row = {c: req.features.get(c, None) for c in _MODEL.feature_cols}
    X = pd.DataFrame([row], columns=_MODEL.feature_cols)
    proba_declined = float(_MODEL.predict_proba(X)[0])
    thr = req.threshold if req.threshold is not None else _MODEL.threshold
    pred = int(proba_declined >= thr)
    return {
        "predicted_class": "Declined" if pred == 1 else "Completed",
        "predicted_label": pred,
        "probability_declined": round(proba_declined, 4),
        "probability_completed": round(1.0 - proba_declined, 4),
        "threshold_used": round(float(thr), 4),
        "explanation": (
            f"Probability of Declined is {proba_declined:.1%}; the decision threshold is "
            f"{thr:.2f}, so the claim is classified as "
            f"{'Declined' if pred else 'Completed'}."
        ),
    }


@app.get("/model-summary")
def model_summary():
    _load()
    if not _META:
        raise HTTPException(503, "Model metadata not available.")
    return {
        "best_model": _META.get("model_type"),
        "stage": _META.get("stage"),
        "imbalance_strategy": _META.get("imbalance_strategy"),
        "best_hyperparameters": _META.get("params"),
        "threshold": _META.get("threshold"),
        "test_metrics": _META.get("test_metrics"),
        "positive_class": _META.get("positive_class"),
    }


@app.get("/model-comparison")
def model_comparison():
    _load()
    return {"leaderboard": _DASH.get("leaderboard", [])}


@app.get("/threshold-analysis")
def threshold_analysis():
    _load()
    return {"threshold_analysis": _DASH.get("threshold_analysis", []),
            "selected_threshold": _META.get("threshold")}


@app.get("/feature-importance")
def feature_importance():
    _load()
    best = _DASH.get("best_model", {})
    return {"model": best.get("model"),
            "feature_importance": best.get("feature_importance", [])}


@app.get("/dashboard")
def dashboard():
    _load()
    if not _DASH:
        raise HTTPException(503, "Dashboard data not available. Run src/run_pipeline.py first.")
    return _DASH
