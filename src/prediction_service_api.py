"""Production model-serving API for the claim-approval model.

A focused inference service (separate from `dashboard_api.py`, which backs the React
dashboard). It loads `models/best_model.joblib` + `best_model_meta.json` once and
exposes operational endpoints with request/response schemas, input validation,
structured error handling and in-memory service metrics.

Run from the project root:
    .venv/bin/uvicorn prediction_service_api:app --app-dir src --port 8001
    # docs: http://localhost:8001/docs

Endpoints:
    GET  /health         liveness — is the server up?
    GET  /ready          readiness — is the model loaded and usable?
    GET  /metadata       model version, training date, metrics, feature importance
    POST /predict        single-sample inference
    POST /predict-batch  multi-sample inference
    GET  /metrics        request count, error rate, latency stats
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

# Make sibling modules importable whether launched via uvicorn or directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import pandas as pd
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

import prompts  # LLM model + prompt config (version-controlled)
from config import APP_VERSION, BEST_MODEL_META, BEST_MODEL_PATH
from model_factory import ClaimModel  # noqa: F401  (needed for joblib unpickling)

SERVICE_NAME = "claim-approval-serving-api"
SERVICE_VERSION = "1.0.0"
MAX_BATCH = 1000

# --------------------------------------------------------------------------
# Model state (loaded once at startup)
# --------------------------------------------------------------------------
class ModelState:
    def __init__(self):
        self.model: ClaimModel | None = None
        self.meta: dict = {}
        self.load_error: str | None = None
        self.loaded_at: str | None = None

    def load(self):
        try:
            if not BEST_MODEL_PATH.exists():
                self.load_error = f"Model artifact not found: {BEST_MODEL_PATH}. Run src/run_pipeline.py."
                return
            self.model = joblib.load(BEST_MODEL_PATH)
            self.meta = json.loads(BEST_MODEL_META.read_text()) if BEST_MODEL_META.exists() else {}
            self.loaded_at = datetime.now(timezone.utc).isoformat()
            self.load_error = None
        except Exception as exc:  # pragma: no cover - defensive
            self.load_error = f"{type(exc).__name__}: {exc}"
            self.model = None

    @property
    def ready(self) -> bool:
        return self.model is not None


STATE = ModelState()


# --------------------------------------------------------------------------
# Service metrics
# --------------------------------------------------------------------------
class Metrics:
    def __init__(self):
        self.lock = Lock()
        self.start = time.time()
        self.total = 0
        self.errors = 0
        self.by_path: dict[str, int] = defaultdict(int)
        self.errors_by_path: dict[str, int] = defaultdict(int)
        self.latencies_ms: deque[float] = deque(maxlen=1000)
        self.predictions = 0  # number of samples scored

    def record(self, path: str, latency_ms: float, is_error: bool):
        with self.lock:
            self.total += 1
            self.by_path[path] += 1
            self.latencies_ms.append(latency_ms)
            if is_error:
                self.errors += 1
                self.errors_by_path[path] += 1

    def add_predictions(self, n: int):
        with self.lock:
            self.predictions += n

    def snapshot(self) -> dict:
        with self.lock:
            lat = sorted(self.latencies_ms)
            n = len(lat)

            def pct(p):
                if not n:
                    return 0.0
                idx = min(n - 1, int(round((p / 100) * (n - 1))))
                return round(lat[idx], 2)

            avg = round(sum(lat) / n, 2) if n else 0.0
            return {
                "request_count": self.total,
                "error_count": self.errors,
                "error_rate": round(self.errors / self.total, 4) if self.total else 0.0,
                "samples_predicted": self.predictions,
                "latency_ms": {
                    "avg": avg,
                    "p50": pct(50),
                    "p95": pct(95),
                    "p99": pct(99),
                    "max": round(max(lat), 2) if n else 0.0,
                    "window": n,
                },
                "requests_by_path": dict(self.by_path),
                "errors_by_path": dict(self.errors_by_path),
                "uptime_seconds": round(time.time() - self.start, 1),
            }


METRICS = Metrics()


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class PredictRequest(BaseModel):
    features: dict = Field(..., description="Raw feature name -> value. Missing features are imputed.",
                           json_schema_extra={"example": {"rrp": 1799, "excessFee": 139,
                                                          "coverage": "ADLD", "country": "SE",
                                                          "claimType": "Accidental Damage"}})
    threshold: float | None = Field(None, ge=0.0, le=1.0,
                                    description="Optional decision-threshold override in [0,1].")

    @field_validator("features")
    @classmethod
    def _non_empty(cls, v):
        if not isinstance(v, dict) or not v:
            raise ValueError("`features` must be a non-empty object.")
        return v


class BatchPredictRequest(BaseModel):
    samples: list[dict] = Field(..., min_length=1, max_length=MAX_BATCH,
                                description=f"List of feature objects (1..{MAX_BATCH}).")
    threshold: float | None = Field(None, ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_label: int
    probability_declined: float
    probability_completed: float
    threshold_used: float
    model_version: str
    explanation: str


class BatchPredictionResponse(BaseModel):
    count: int
    n_declined: int
    n_completed: int
    threshold_used: float
    model_version: str
    predictions: list[PredictionResponse]


# --------------------------------------------------------------------------
# App + middleware
# --------------------------------------------------------------------------
app = FastAPI(title="Claim Approval Serving API", version=SERVICE_VERSION,
              description="Inference service for the claim-approval (Declined vs Completed) model.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    is_error = False
    try:
        response = await call_next(request)
        # Count any failed request (4xx client + 5xx server) toward the error rate.
        is_error = response.status_code >= 400
        return response
    except Exception:
        is_error = True
        raise
    finally:
        METRICS.record(request.url.path, (time.perf_counter() - t0) * 1000.0, is_error)


@app.on_event("startup")
def _startup():
    STATE.load()


def _error(status_code: int, code: str, message: str, details=None) -> JSONResponse:
    payload = {"error": {"code": code, "message": message}}
    if details is not None:
        payload["error"]["details"] = details
    return JSONResponse(status_code=status_code, content=payload)


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    return _error(status.HTTP_422_UNPROCESSABLE_ENTITY, "validation_error",
                  "Request body failed validation.", details=exc.errors())


@app.exception_handler(Exception)
async def _unhandled_handler(request: Request, exc: Exception):  # pragma: no cover
    return _error(status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error",
                  "An unexpected error occurred.", details=f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# Validation + inference helpers
# --------------------------------------------------------------------------
def validate_features(features: dict) -> list[str]:
    """Return a list of human-readable validation errors (empty = valid)."""
    errs: list[str] = []
    feature_cols = set(STATE.model.feature_cols)
    num_cols = set(STATE.model.num_cols)

    unknown = [k for k in features if k not in feature_cols]
    if unknown:
        errs.append(f"Unknown feature(s): {sorted(unknown)}. "
                    f"Allowed: {sorted(feature_cols)}.")
    for k, v in features.items():
        if k in num_cols and v is not None and isinstance(v, bool):
            errs.append(f"Numeric feature '{k}' must be a number, got boolean.")
        elif k in num_cols and v is not None and not isinstance(v, (int, float)):
            errs.append(f"Numeric feature '{k}' must be a number or null, got {type(v).__name__}.")
    return errs


def _score_frame(samples: list[dict], threshold: float | None):
    """Vectorized scoring of N raw feature dicts -> list of PredictionResponse dicts."""
    cols = STATE.model.feature_cols
    rows = [{c: s.get(c, None) for c in cols} for s in samples]
    X = pd.DataFrame(rows, columns=cols)
    proba = STATE.model.predict_proba(X)
    thr = STATE.model.threshold if threshold is None else float(threshold)
    version = STATE.meta.get("model_version", "unknown")
    out = []
    for p in proba:
        p = float(p)
        label = int(p >= thr)
        out.append({
            "predicted_class": "Declined" if label else "Completed",
            "predicted_label": label,
            "probability_declined": round(p, 4),
            "probability_completed": round(1.0 - p, 4),
            "threshold_used": round(thr, 4),
            "model_version": version,
            "explanation": (f"P(Declined)={p:.1%} vs threshold {thr:.2f} "
                            f"=> {'Declined' if label else 'Completed'}."),
        })
    return out, thr, version


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/health", tags=["ops"])
def health():
    """Liveness probe — the process is up and serving."""
    return {"status": "ok", "service": SERVICE_NAME, "version": SERVICE_VERSION,
            "time": datetime.now(timezone.utc).isoformat()}


@app.get("/ready", tags=["ops"])
def ready():
    """Readiness probe — the model is loaded and inference can be served."""
    if STATE.ready:
        return {"ready": True, "model_loaded": True,
                "model_version": STATE.meta.get("model_version"),
                "loaded_at": STATE.loaded_at}
    return _error(status.HTTP_503_SERVICE_UNAVAILABLE, "not_ready",
                  "Model is not loaded.", details=STATE.load_error)


@app.get("/metadata", tags=["model"])
def metadata():
    """Model metadata: version, training date, hyperparameters, metrics, feature importance."""
    if not STATE.ready:
        return _error(status.HTTP_503_SERVICE_UNAVAILABLE, "not_ready",
                      "Model is not loaded.", details=STATE.load_error)
    m = STATE.meta
    return {
        "app_version": APP_VERSION,
        "llm": {"model": prompts.MODEL, "config_version": prompts.VERSION},
        "model_name": m.get("model_type"),
        "model_version": m.get("model_version"),
        "training_date": m.get("trained_at"),
        "stage": m.get("stage"),
        "imbalance_strategy": m.get("imbalance_strategy"),
        "decision_threshold": m.get("threshold"),
        "positive_class": m.get("positive_class", "Declined"),
        "negative_class": m.get("negative_class", "Completed"),
        "hyperparameters": m.get("params", {}),
        "optuna_trials": m.get("optuna_trials"),
        "n_features": len(STATE.model.feature_cols),
        "feature_columns": STATE.model.feature_cols,
        "numeric_features": STATE.model.num_cols,
        "categorical_features": STATE.model.cat_cols,
        "feature_importance": m.get("feature_importance", []),
        "test_metrics": m.get("test_metrics", {}),
    }


_EXPLANATION: dict = {}  # cache so the LLM is called at most once per process


@app.post("/explain-prediction", tags=["model"])
def explain_prediction(req: PredictRequest):
    """Claims-adjuster-facing LLM explanation of THIS claim's model prediction.

    The model makes the prediction (recomputed here); the LLM only explains it to
    support manual review — it never decides."""
    if not STATE.ready:
        return _error(status.HTTP_503_SERVICE_UNAVAILABLE, "not_ready",
                      "Model is not loaded.", details=STATE.load_error)
    from llm_explain import explanation_available, generate_prediction_explanation

    if not explanation_available():
        return _error(status.HTTP_503_SERVICE_UNAVAILABLE, "llm_disabled",
                      "OPENAI_API_KEY is not set on the server, so AI explanations are disabled.")
    errs = validate_features(req.features)
    if errs:
        return _error(status.HTTP_422_UNPROCESSABLE_ENTITY, "invalid_features",
                      "Input feature validation failed.", details=errs)
    preds, _, _ = _score_frame([req.features], req.threshold)
    prediction = preds[0]
    m = STATE.meta
    model_info = {
        "model_name": m.get("model_type"), "stage": m.get("stage"),
        "model_version": m.get("model_version"), "imbalance_strategy": m.get("imbalance_strategy"),
        "threshold": m.get("threshold"), "test_metrics": m.get("test_metrics", {}),
    }
    try:
        text = generate_prediction_explanation(prediction, req.features, model_info,
                                               m.get("feature_importance", []))
    except Exception as exc:
        return _error(status.HTTP_502_BAD_GATEWAY, "llm_error",
                      "LLM explanation failed.", details=f"{type(exc).__name__}: {exc}")
    return {"prediction": prediction, "explanation": text}


@app.post("/explain-prediction-customer", tags=["model"])
def explain_prediction_customer(req: PredictRequest):
    """Customer-facing, plain-language LLM explanation of THIS claim's prediction.

    The model makes the prediction (recomputed here); the LLM only explains it in
    simple terms for the customer — it never decides."""
    if not STATE.ready:
        return _error(status.HTTP_503_SERVICE_UNAVAILABLE, "not_ready",
                      "Model is not loaded.", details=STATE.load_error)
    from llm_explain import explanation_available, generate_customer_explanation

    if not explanation_available():
        return _error(status.HTTP_503_SERVICE_UNAVAILABLE, "llm_disabled",
                      "OPENAI_API_KEY is not set on the server, so AI explanations are disabled.")
    errs = validate_features(req.features)
    if errs:
        return _error(status.HTTP_422_UNPROCESSABLE_ENTITY, "invalid_features",
                      "Input feature validation failed.", details=errs)
    preds, _, _ = _score_frame([req.features], req.threshold)
    prediction = preds[0]
    m = STATE.meta
    model_info = {
        "model_name": m.get("model_type"), "stage": m.get("stage"),
        "model_version": m.get("model_version"), "imbalance_strategy": m.get("imbalance_strategy"),
        "threshold": m.get("threshold"), "test_metrics": m.get("test_metrics", {}),
    }
    try:
        text = generate_customer_explanation(prediction, req.features, model_info,
                                             m.get("feature_importance", []))
    except Exception as exc:
        return _error(status.HTTP_502_BAD_GATEWAY, "llm_error",
                      "LLM explanation failed.", details=f"{type(exc).__name__}: {exc}")
    return {"prediction": prediction, "explanation": text}


@app.get("/explain", tags=["model"])
def explain(refresh: bool = False):
    """LLM-generated (gpt-4o-mini via LangChain) plain-English explanation of the
    serving model and its feature importance. Cached after first generation."""
    if not STATE.ready:
        return _error(status.HTTP_503_SERVICE_UNAVAILABLE, "not_ready",
                      "Model is not loaded.", details=STATE.load_error)
    from llm_explain import explanation_available, generate_model_explanation

    if not explanation_available():
        return _error(status.HTTP_503_SERVICE_UNAVAILABLE, "llm_disabled",
                      "OPENAI_API_KEY is not set on the server, so AI explanations are disabled.")
    if _EXPLANATION.get("text") and not refresh:
        return {"explanation": _EXPLANATION["text"], "cached": True,
                "model": STATE.meta.get("model_type"),
                "model_version": STATE.meta.get("model_version")}
    m = STATE.meta
    model_info = {
        "model_name": m.get("model_type"), "stage": m.get("stage"),
        "model_version": m.get("model_version"), "imbalance_strategy": m.get("imbalance_strategy"),
        "threshold": m.get("threshold"), "test_metrics": m.get("test_metrics", {}),
    }
    try:
        text = generate_model_explanation(model_info, m.get("feature_importance", []))
    except Exception as exc:
        return _error(status.HTTP_502_BAD_GATEWAY, "llm_error",
                      "LLM explanation failed.", details=f"{type(exc).__name__}: {exc}")
    _EXPLANATION["text"] = text
    return {"explanation": text, "cached": False,
            "model": m.get("model_type"), "model_version": m.get("model_version")}


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(req: PredictRequest):
    """Single-sample inference."""
    if not STATE.ready:
        return _error(status.HTTP_503_SERVICE_UNAVAILABLE, "not_ready",
                      "Model is not loaded.", details=STATE.load_error)
    errs = validate_features(req.features)
    if errs:
        return _error(status.HTTP_422_UNPROCESSABLE_ENTITY, "invalid_features",
                      "Input feature validation failed.", details=errs)
    preds, _, _ = _score_frame([req.features], req.threshold)
    METRICS.add_predictions(1)
    return preds[0]


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["inference"])
def predict_batch(req: BatchPredictRequest):
    """Batch inference for multiple samples (vectorized)."""
    if not STATE.ready:
        return _error(status.HTTP_503_SERVICE_UNAVAILABLE, "not_ready",
                      "Model is not loaded.", details=STATE.load_error)
    all_errs = {}
    for i, s in enumerate(req.samples):
        if not isinstance(s, dict) or not s:
            all_errs[i] = ["sample must be a non-empty object"]
            continue
        e = validate_features(s)
        if e:
            all_errs[i] = e
    if all_errs:
        return _error(status.HTTP_422_UNPROCESSABLE_ENTITY, "invalid_features",
                      "One or more samples failed validation.", details=all_errs)
    preds, thr, version = _score_frame(req.samples, req.threshold)
    METRICS.add_predictions(len(preds))
    n_declined = sum(p["predicted_label"] for p in preds)
    return {
        "count": len(preds),
        "n_declined": n_declined,
        "n_completed": len(preds) - n_declined,
        "threshold_used": round(thr, 4),
        "model_version": version,
        "predictions": preds,
    }


@app.get("/metrics", tags=["ops"])
def metrics():
    """Basic service metrics: request count, error rate, latency stats."""
    return METRICS.snapshot()
