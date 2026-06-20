"""Build, fit and wrap the four model families behind one interface.

`ClaimModel` bundles the fitted preprocessor (if any) with the estimator and the
chosen decision threshold so that prediction on raw feature rows is identical at
train time, evaluation time and serving time. It is what gets pickled as the
final model artifact and what the FastAPI service loads.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from config import IMBALANCE_WEIGHT, RANDOM_STATE
from preprocessing import (
    build_tree_preprocessor,
    catboost_cat_indices,
    prepare_catboost_frame,
)

TREE_MODELS = {"RandomForest", "XGBoost", "LightGBM"}
EARLY_STOP_ROUNDS = 50


@dataclass
class ClaimModel:
    """Fitted model + preprocessing + threshold, with a uniform predict API."""

    model_type: str
    estimator: Any
    num_cols: list[str]
    cat_cols: list[str]
    feature_cols: list[str]
    imbalance_strategy: str
    params: dict = field(default_factory=dict)
    preprocessor: Any = None          # ColumnTransformer for tree models, else None
    threshold: float = 0.5

    # --- prediction ------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of the positive class (Declined = 1)."""
        X = X[self.feature_cols] if all(c in X.columns for c in self.feature_cols) else X
        if self.model_type == "CatBoost":
            Xc = prepare_catboost_frame(X, self.cat_cols)[self.feature_cols]
            return self.estimator.predict_proba(Xc)[:, 1]
        Xt = self.preprocessor.transform(X)
        return self.estimator.predict_proba(Xt)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        t = self.threshold if threshold is None else threshold
        return (self.predict_proba(X) >= t).astype(int)

    # --- explainability --------------------------------------------------
    def feature_importance(self) -> dict[str, float]:
        if self.model_type == "CatBoost":
            names = self.feature_cols
            imp = self.estimator.get_feature_importance()
        else:
            names = list(self.preprocessor.get_feature_names_out())
            imp = getattr(self.estimator, "feature_importances_", None)
            if imp is None:
                return {}
        return {str(n): float(v) for n, v in zip(names, imp)}


# --- estimator construction ---------------------------------------------
def _rf(params, imbalance):
    from sklearn.ensemble import RandomForestClassifier

    cw = params.pop("class_weight", imbalance)
    if cw == "custom":
        cw = {0: 1.0, 1: IMBALANCE_WEIGHT}
    return RandomForestClassifier(
        random_state=RANDOM_STATE, n_jobs=-1, class_weight=cw, **params
    )


def _xgb(params, imbalance):
    from xgboost import XGBClassifier

    spw = params.pop("scale_pos_weight", imbalance)
    return XGBClassifier(
        random_state=RANDOM_STATE, n_jobs=-1, eval_metric="aucpr",
        early_stopping_rounds=EARLY_STOP_ROUNDS, scale_pos_weight=spw,
        tree_method="hist", **params,
    )


def _lgbm(params, imbalance):
    from lightgbm import LGBMClassifier

    spw = params.pop("scale_pos_weight", imbalance)
    return LGBMClassifier(
        random_state=RANDOM_STATE, n_jobs=-1, scale_pos_weight=spw,
        verbose=-1, **params,
    )


def _catboost(params, imbalance):
    from catboost import CatBoostClassifier

    kwargs = dict(
        random_seed=RANDOM_STATE, eval_metric="AUC", verbose=0,
        early_stopping_rounds=EARLY_STOP_ROUNDS,
    )
    if imbalance == "auto_class_weights":
        kwargs["auto_class_weights"] = "Balanced"
    else:  # explicit class weights {Completed:1, Declined:5.36}
        kwargs["class_weights"] = [1.0, IMBALANCE_WEIGHT]
    kwargs.update(params)
    return CatBoostClassifier(**kwargs)


def fit_model(model_type: str, params: dict, imbalance_strategy: str, ds) -> ClaimModel:
    """Fit a model family on the training split and return a `ClaimModel`.

    Boosting models use the validation split for early stopping. Random Forest
    has no early stopping. The validation set is *only* used to stop training /
    select hyper-parameters -- never mixed into training data.
    """
    params = dict(params)  # copy; constructors pop from it

    if model_type == "RandomForest":
        pre = build_tree_preprocessor(ds.num_cols, ds.cat_cols)
        Xtr = pre.fit_transform(ds.X_train)
        est = _rf(params, imbalance_strategy)
        est.fit(Xtr, ds.y_train)
        return ClaimModel("RandomForest", est, ds.num_cols, ds.cat_cols, ds.feature_cols,
                          imbalance_strategy, params, preprocessor=pre)

    if model_type == "XGBoost":
        pre = build_tree_preprocessor(ds.num_cols, ds.cat_cols)
        Xtr = pre.fit_transform(ds.X_train)
        Xva = pre.transform(ds.X_val)
        est = _xgb(params, imbalance_strategy)
        est.fit(Xtr, ds.y_train, eval_set=[(Xva, ds.y_val)], verbose=False)
        return ClaimModel("XGBoost", est, ds.num_cols, ds.cat_cols, ds.feature_cols,
                          imbalance_strategy, params, preprocessor=pre)

    if model_type == "LightGBM":
        import lightgbm as lgb

        pre = build_tree_preprocessor(ds.num_cols, ds.cat_cols)
        Xtr = pre.fit_transform(ds.X_train)
        Xva = pre.transform(ds.X_val)
        est = _lgbm(params, imbalance_strategy)
        est.fit(Xtr, ds.y_train, eval_set=[(Xva, ds.y_val)], eval_metric="auc",
                callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                           lgb.log_evaluation(0)])
        return ClaimModel("LightGBM", est, ds.num_cols, ds.cat_cols, ds.feature_cols,
                          imbalance_strategy, params, preprocessor=pre)

    if model_type == "CatBoost":
        cat_idx = catboost_cat_indices(ds.feature_cols, ds.cat_cols)
        Xtr = prepare_catboost_frame(ds.X_train, ds.cat_cols)[ds.feature_cols]
        Xva = prepare_catboost_frame(ds.X_val, ds.cat_cols)[ds.feature_cols]
        est = _catboost(params, imbalance_strategy)
        est.set_params(cat_features=cat_idx)
        est.fit(Xtr, ds.y_train, eval_set=(Xva, ds.y_val), use_best_model=True)
        return ClaimModel("CatBoost", est, ds.num_cols, ds.cat_cols, ds.feature_cols,
                          imbalance_strategy, params, preprocessor=None)

    raise ValueError(f"Unknown model_type: {model_type}")


# Default imbalance strategy per family (used by baselines).
DEFAULT_IMBALANCE = {
    "RandomForest": "balanced",
    "XGBoost": IMBALANCE_WEIGHT,
    "LightGBM": IMBALANCE_WEIGHT,
    "CatBoost": "class_weights",
}

IMBALANCE_LABEL = {
    "RandomForest": "class_weight='balanced'",
    "XGBoost": f"scale_pos_weight={IMBALANCE_WEIGHT}",
    "LightGBM": f"scale_pos_weight={IMBALANCE_WEIGHT}",
    "CatBoost": "class_weights={Completed:1, Declined:5.36}",
}
