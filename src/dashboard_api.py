"""FastAPI service exposing the trained model and the dashboard data.

Run from the project root:
    .venv/bin/uvicorn dashboard_api:app --app-dir src --reload --port 8000

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
_EXPLANATION: dict = {}  # cache: {"text": str} so we call the LLM at most once per process


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
                      "/threshold-analysis", "/feature-importance", "/explain",
                      "/explain-prediction", "/explain-prediction-customer",
                      "/predictions/recent", "/predictions/{id}/explanation",
                      "/predictions/{id}/decision", "/dashboard"],
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
    result = {
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
    # Best-effort: persist the features + prediction to RDS (never breaks /predict).
    import db
    pid = db.save_prediction(
        _MODEL, row, {**result, "model_version": _META.get("model_version")})
    result["saved_to_db"] = pid is not None
    result["prediction_id"] = pid
    return result


@app.get("/predictions/recent")
def predictions_recent(limit: int = 20):
    """Most recent predictions saved to the database (newest first).

    Returns enabled=false when the DB is not configured."""
    import db
    rows = db.recent_predictions(limit)
    if rows is None:
        return {"enabled": False, "count": 0, "predictions": []}
    return {"enabled": True, "count": len(rows), "predictions": rows}


class ExplanationSaveRequest(BaseModel):
    kind: str = Field(..., description="'adjuster' or 'customer'")
    explanation: str = Field(..., description="The LLM explanation text to store")


@app.post("/predictions/{prediction_id}/explanation")
def save_prediction_explanation(prediction_id: int, req: ExplanationSaveRequest):
    """Attach an LLM explanation (adjuster or customer) to a saved prediction row."""
    import db
    if not db.db_enabled():
        raise HTTPException(503, "Database is not configured on the server.")
    if req.kind not in ("adjuster", "customer"):
        raise HTTPException(422, "kind must be 'adjuster' or 'customer'.")
    res = db.save_explanation(prediction_id, req.kind, req.explanation)
    if not res["ok"]:
        if res["reason"] == "not_found":
            raise HTTPException(404, f"No saved prediction with id {prediction_id}.")
        raise HTTPException(502, f"Could not save explanation ({res['reason']}).")
    return {"saved": True, "prediction_id": prediction_id, "kind": req.kind}


class DecisionSaveRequest(BaseModel):
    decision: str = Field(..., description="'Completed' (approve) or 'Declined' (decline)")


@app.post("/predictions/{prediction_id}/decision")
def save_prediction_decision(prediction_id: int, req: DecisionSaveRequest):
    """Record the claims adjuster's final decision on a saved prediction row.

    The model's predicted_class is preserved; this stores the human decision in
    the adjuster_decision column."""
    import db
    if not db.db_enabled():
        raise HTTPException(503, "Database is not configured on the server.")
    if req.decision not in ("Completed", "Declined"):
        raise HTTPException(422, "decision must be 'Completed' or 'Declined'.")
    res = db.save_decision(prediction_id, req.decision)
    if not res["ok"]:
        if res["reason"] == "not_found":
            raise HTTPException(404, f"No saved prediction with id {prediction_id}.")
        raise HTTPException(502, f"Could not save decision ({res['reason']}).")
    return {"saved": True, "prediction_id": prediction_id, "decision": req.decision}


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


@app.post("/explain-prediction")
def explain_prediction(req: PredictRequest):
    """Claims-adjuster-facing LLM explanation of THIS claim's model prediction.

    The model makes the prediction (recomputed here from the claim features); the
    LLM only explains that output to support manual review — it never decides."""
    _load()
    if _MODEL is None:
        raise HTTPException(503, "Model not trained yet. Run src/run_pipeline.py first.")
    from llm_explain import explanation_available, generate_prediction_explanation

    if not explanation_available():
        raise HTTPException(
            503, "OPENAI_API_KEY is not set on the server, so AI explanations are disabled.")

    row = {c: req.features.get(c, None) for c in _MODEL.feature_cols}
    X = pd.DataFrame([row], columns=_MODEL.feature_cols)
    proba = float(_MODEL.predict_proba(X)[0])
    thr = req.threshold if req.threshold is not None else _MODEL.threshold
    pred = int(proba >= thr)
    prediction = {
        "predicted_class": "Declined" if pred else "Completed",
        "predicted_label": pred,
        "probability_declined": round(proba, 4),
        "probability_completed": round(1.0 - proba, 4),
        "threshold_used": round(float(thr), 4),
    }
    model_info = {
        "model_name": _META.get("model_type"), "stage": _META.get("stage"),
        "model_version": _META.get("model_version"),
        "imbalance_strategy": _META.get("imbalance_strategy"),
        "threshold": _META.get("threshold"), "test_metrics": _META.get("test_metrics", {}),
    }
    fi = _META.get("feature_importance") or _DASH.get("best_model", {}).get("feature_importance", [])
    try:
        text = generate_prediction_explanation(prediction, req.features, model_info, fi)
    except Exception as exc:
        raise HTTPException(502, f"LLM explanation failed: {type(exc).__name__}: {exc}")
    return {"prediction": prediction, "explanation": text}


@app.post("/explain-prediction-customer")
def explain_prediction_customer(req: PredictRequest):
    """Customer-facing, plain-language LLM explanation of THIS claim's prediction.

    The model makes the prediction (recomputed here); the LLM only explains it in
    simple terms for the customer — it never decides."""
    _load()
    if _MODEL is None:
        raise HTTPException(503, "Model not trained yet. Run src/run_pipeline.py first.")
    from llm_explain import explanation_available, generate_customer_explanation

    if not explanation_available():
        raise HTTPException(
            503, "OPENAI_API_KEY is not set on the server, so AI explanations are disabled.")

    row = {c: req.features.get(c, None) for c in _MODEL.feature_cols}
    X = pd.DataFrame([row], columns=_MODEL.feature_cols)
    proba = float(_MODEL.predict_proba(X)[0])
    thr = req.threshold if req.threshold is not None else _MODEL.threshold
    pred = int(proba >= thr)
    prediction = {
        "predicted_class": "Declined" if pred else "Completed",
        "predicted_label": pred,
        "probability_declined": round(proba, 4),
        "probability_completed": round(1.0 - proba, 4),
        "threshold_used": round(float(thr), 4),
    }
    model_info = {
        "model_name": _META.get("model_type"), "stage": _META.get("stage"),
        "model_version": _META.get("model_version"),
        "imbalance_strategy": _META.get("imbalance_strategy"),
        "threshold": _META.get("threshold"), "test_metrics": _META.get("test_metrics", {}),
    }
    fi = _META.get("feature_importance") or _DASH.get("best_model", {}).get("feature_importance", [])
    try:
        text = generate_customer_explanation(prediction, req.features, model_info, fi)
    except Exception as exc:
        raise HTTPException(502, f"LLM explanation failed: {type(exc).__name__}: {exc}")
    return {"prediction": prediction, "explanation": text}


@app.get("/explain")
def explain(refresh: bool = False):
    """LLM-generated (gpt-4o-mini via LangChain) plain-English explanation of the
    serving model and its feature importance. Cached after first generation."""
    _load()
    if _MODEL is None:
        raise HTTPException(503, "Model not trained yet. Run src/run_pipeline.py first.")

    from llm_explain import explanation_available, generate_model_explanation

    if not explanation_available():
        raise HTTPException(
            503, "OPENAI_API_KEY is not set on the server, so AI explanations are disabled.")

    if _EXPLANATION.get("text") and not refresh:
        return {"explanation": _EXPLANATION["text"], "cached": True,
                "model": _META.get("model_type"), "model_version": _META.get("model_version")}

    model_info = {
        "model_name": _META.get("model_type"),
        "stage": _META.get("stage"),
        "model_version": _META.get("model_version"),
        "imbalance_strategy": _META.get("imbalance_strategy"),
        "threshold": _META.get("threshold"),
        "test_metrics": _META.get("test_metrics", {}),
    }
    fi = _META.get("feature_importance") or _DASH.get("best_model", {}).get("feature_importance", [])
    try:
        text = generate_model_explanation(model_info, fi)
    except Exception as exc:
        raise HTTPException(502, f"LLM explanation failed: {type(exc).__name__}: {exc}")
    _EXPLANATION["text"] = text
    return {"explanation": text, "cached": False,
            "model": _META.get("model_type"), "model_version": _META.get("model_version")}


@app.get("/dashboard")
def dashboard():
    _load()
    if not _DASH:
        raise HTTPException(503, "Dashboard data not available. Run src/run_pipeline.py first.")
    return _DASH
