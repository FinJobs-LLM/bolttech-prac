"""Baseline (pre-optimization) configurations for the four model families.

Each baseline uses sensible defaults plus the imbalance strategy from the
project brief, so the optimized models have a fair reference point to beat.
"""
from __future__ import annotations

from config import IMBALANCE_WEIGHT
from ml.model_factory import DEFAULT_IMBALANCE, fit_model

BASELINE_PARAMS = {
    "RandomForest": {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 2},
    "XGBoost": {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.1,
                "subsample": 0.9, "colsample_bytree": 0.9},
    "LightGBM": {"n_estimators": 400, "num_leaves": 31, "max_depth": -1,
                 "learning_rate": 0.1, "subsample": 0.9, "colsample_bytree": 0.9},
    "CatBoost": {"iterations": 400, "depth": 6, "learning_rate": 0.1, "l2_leaf_reg": 3.0},
}


def build_baseline(model_type: str, ds):
    """Fit the baseline `ClaimModel` for one family."""
    return fit_model(model_type, dict(BASELINE_PARAMS[model_type]),
                     DEFAULT_IMBALANCE[model_type], ds)
