"""Thin helpers around MLflow so logging is consistent across the pipeline."""
from __future__ import annotations

import os
from pathlib import Path

# MLflow 3.x treats the filesystem store as "maintenance mode" and errors unless
# we opt in. A local file store keeps the project self-contained (no DB server)
# and `mlflow ui --backend-store-uri ./mlruns` works directly.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow

from config import MLFLOW_EXPERIMENT, MLRUNS_DIR


def init_mlflow() -> None:
    """Point MLflow at the local ./mlruns store and select the experiment."""
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLRUNS_DIR.resolve().as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def log_metrics_prefixed(metrics: dict, prefix: str) -> None:
    """Log numeric metrics, skipping nested structures (e.g. confusion matrix)."""
    flat = {
        f"{prefix}_{k}": float(v)
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    if flat:
        mlflow.log_metrics(flat)


def log_figure(path: str | Path) -> None:
    p = Path(path)
    if p.exists():
        mlflow.log_artifact(str(p), artifact_path="figures")
