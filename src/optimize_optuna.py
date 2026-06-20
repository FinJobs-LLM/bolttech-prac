"""Optuna hyper-parameter search, one study per model family.

Objective: **maximise PR-AUC on the validation set** -- the right target for an
imbalanced problem where we care about ranking the minority `Declined` class
well across thresholds, not raw accuracy.

MLflow structure: a parent run per family ("XGBoost_optuna") with one nested
child run per trial recording that trial's params and validation PR-AUC.
"""
from __future__ import annotations

import warnings

import mlflow
import optuna
from sklearn.metrics import average_precision_score

from config import IMBALANCE_WEIGHT, RANDOM_STATE
from model_factory import fit_model

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


def suggest_params(model_type: str, trial: optuna.Trial):
    """Return (params, imbalance_strategy) sampled for this trial."""
    if model_type == "RandomForest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 24),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7, 1.0]),
            "class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", "balanced_subsample", "custom"]),
        }
        return params, params["class_weight"]

    if model_type == "XGBoost":
        spw = trial.suggest_float("scale_pos_weight", 1.0, 12.0)
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "scale_pos_weight": spw,
        }
        return params, f"scale_pos_weight={spw:.2f}"

    if model_type == "LightGBM":
        spw = trial.suggest_float("scale_pos_weight", 1.0, 12.0)
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=50),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "scale_pos_weight": spw,
        }
        return params, f"scale_pos_weight={spw:.2f}"

    if model_type == "CatBoost":
        imb = trial.suggest_categorical("imbalance", ["class_weights", "auto_class_weights"])
        params = {
            "iterations": trial.suggest_int("iterations", 200, 900, step=50),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
        }
        return params, imb

    raise ValueError(model_type)


def optimize_model(model_type: str, ds, n_trials: int, log_mlflow: bool = True) -> dict:
    """Run an Optuna study and return the best params + search history."""

    def objective(trial: optuna.Trial) -> float:
        params, imbalance = suggest_params(model_type, trial)
        model = fit_model(model_type, params, imbalance, ds)
        proba = model.predict_proba(ds.X_val)
        pr_auc = float(average_precision_score(ds.y_val, proba))
        if log_mlflow:
            with mlflow.start_run(run_name=f"{model_type}_trial_{trial.number}", nested=True):
                mlflow.set_tag("stage", "optuna_trial")
                mlflow.set_tag("model", model_type)
                mlflow.log_params({k: v for k, v in trial.params.items()})
                mlflow.log_metric("val_pr_auc", pr_auc)
        return pr_auc

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name=f"{model_type}_pr_auc")

    parent_cm = (mlflow.start_run(run_name=f"{model_type}_optuna")
                 if log_mlflow else _NullCtx())
    with parent_cm:
        if log_mlflow:
            mlflow.set_tag("stage", "optuna_search")
            mlflow.set_tag("model", model_type)
        study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
        if log_mlflow:
            mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
            mlflow.log_metric("best_val_pr_auc", study.best_value)

    # Re-derive params/imbalance for the best trial deterministically.
    best_trial = study.best_trial
    best_params, best_imbalance = suggest_params(model_type, _FixedTrial(best_trial.params))

    # Search history + parameter importances for the dashboard.
    history = [
        {"trial": t.number, "value": (t.value if t.value is not None else None),
         "params": t.params}
        for t in study.trials
    ]
    try:
        importances = {k: float(v) for k, v in
                       optuna.importance.get_param_importances(study).items()}
    except Exception:
        importances = {}

    return {
        "model_type": model_type,
        "best_params": best_params,
        "best_imbalance": best_imbalance,
        "best_value": float(study.best_value),
        "n_trials": len(study.trials),
        "history": history,
        "param_importances": importances,
    }


class _NullCtx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


class _FixedTrial:
    """Replays recorded params through suggest_params without sampling."""

    def __init__(self, params: dict):
        self.params = dict(params)

    def suggest_int(self, name, *a, **k): return self.params[name]
    def suggest_float(self, name, *a, **k): return self.params[name]
    def suggest_categorical(self, name, *a, **k): return self.params[name]
