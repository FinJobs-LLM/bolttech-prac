"""Backward-compatibility shim — the implementation now lives in ``ml.model_factory``.

Kept at the top level of ``src`` so that:
* ``models/best_model.joblib`` (pickled as ``model_factory.ClaimModel``) still
  unpickles, and
* existing callers that do ``from model_factory import ClaimModel`` (e.g. the
  FastAPI serving apps) keep working unchanged.

``model_factory.ClaimModel`` is the *same* class object as
``ml.model_factory.ClaimModel``, so old and new model artifacts both load.
"""
from ml.model_factory import *  # noqa: F401,F403  (re-export public API)
from ml.model_factory import (  # noqa: F401  (explicit for pickle / known callers)
    ClaimModel,
    DEFAULT_IMBALANCE,
    IMBALANCE_LABEL,
    fit_model,
)
