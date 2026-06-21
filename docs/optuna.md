# Optuna in this repository

Optuna is the **hyperparameter optimization engine**. Its job is to search each model family's
hyperparameter space *systematically and reproducibly* and return the configuration that best
separates the rare `Declined` class — measured by **validation PR-AUC**, the right objective for an
imbalanced problem.

It does **not** decide the final winner across families (that is `run_pipeline.py`, by validation
PR-AUC), tune the decision threshold (that is `threshold_tuning.py`), or record results (that is
[MLflow](./mlflow.md)). Optuna only *proposes hyperparameters and scores them*.

## Where it lives in the code

All Optuna logic is in **`src/optimize_optuna.py`**:

| Function | Role |
|---|---|
| `suggest_params(model_type, trial)` | Defines the search space per family; returns `(params, imbalance_strategy)`. |
| `optimize_model(model_type, ds, n_trials)` | Creates the study, runs the objective, returns best params + search history. |
| `_FixedTrial` | Replays the best trial's recorded params back through `suggest_params` so the winning config is rebuilt deterministically. |

Driven from `run_pipeline.py`, which calls `optimize_model` for each family, refits the best params
via `model_factory.fit_model`, then hands off to threshold tuning and evaluation.

## The objective

```python
# objective(trial):  fit on train, score PR-AUC on validation
model = fit_model(model_type, params, imbalance, ds)
proba = model.predict_proba(ds.X_val)
return average_precision_score(ds.y_val, proba)   # PR-AUC on the validation set
```

- **Direction:** `maximize`.
- **Metric:** PR-AUC (average precision) of the **positive class `Declined`** on the **validation**
  split. Accuracy is deliberately *not* used — see the project's metric rationale.
- The **test set is never seen** during the search; only train (fit) and validation (score) are used.
- Boosting models additionally use the validation split for **early stopping** inside `fit_model`.

## Search spaces (`suggest_params`)

Each family tunes the hyperparameters called for in the brief:

- **RandomForest:** `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`,
  `max_features`, and `class_weight` ∈ {`balanced`, `balanced_subsample`, `custom`}.
- **XGBoost:** `n_estimators`, `max_depth`, `learning_rate` (log), `subsample`, `colsample_bytree`,
  `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`, `scale_pos_weight`.
- **LightGBM:** `n_estimators`, `num_leaves`, `max_depth`, `learning_rate` (log), `subsample`,
  `colsample_bytree`, `min_child_samples`, `reg_alpha`, `reg_lambda`, `scale_pos_weight`.
- **CatBoost:** `iterations`, `depth`, `learning_rate` (log), `l2_leaf_reg`, `bagging_temperature`,
  `random_strength`, `border_count`, and the imbalance strategy ∈ {`class_weights`,
  `auto_class_weights`}.

The imbalance handling is part of the search: tree models tune `scale_pos_weight` / `class_weight`,
and CatBoost chooses between explicit class weights and `auto_class_weights="Balanced"`.

## Reproducibility

- **Sampler:** `TPESampler(seed=42)` (`config.RANDOM_STATE`) — Tree-structured Parzen Estimator, a
  Bayesian sampler that focuses trials on promising regions.
- **`n_jobs=1`** for the study, so the seeded trial sequence (and the MLflow nesting) is deterministic.
  The models themselves still train multi-threaded internally.
- Same seed + same `N_TRIALS` ⇒ identical search, identical best params.

## Number of trials

`config.N_TRIALS` (default **50**, overridable via the `N_TRIALS` env var):

```bash
python src/run_pipeline.py            # 50 trials per family (default)
N_TRIALS=10 python src/run_pipeline.py  # quick smoke run
```

> **Why 50?** Empirically the seeded TPE search converges by ~30 trials on this dataset, and 50
> reproduces that optimum with headroom. Pushing to 100 raised *validation* PR-AUC but **overfit the
> small validation set** (only 68 `Declined` rows), lowering *test* PR-AUC — so more trials is not
> automatically better here. See the commit history / `reports/final_model_report.md`.

## Outputs of a study

`optimize_model` returns a dict consumed downstream:

- `best_params`, `best_imbalance` — refit by `run_pipeline.py` into the final `ClaimModel`.
- `best_value` — best validation PR-AUC found.
- `history` — per-trial `{trial, value, params}` (powers the dashboard's search-history chart).
- `param_importances` — from `optuna.importance.get_param_importances` (fANOVA); shows which
  hyperparameters mattered most. Powers the dashboard's parameter-importance bar chart.

Per-trial params and scores are also logged to MLflow as nested runs — see [mlflow.md](./mlflow.md).

## End-to-end flow

```
for each family:
  optimize_model ─ TPE search, maximize validation PR-AUC ─► best_params
       │ (each trial logged to MLflow as a nested run)
       ▼
  fit_model(best_params) ─► threshold_tuning (val) ─► evaluate on test
                                   │
run_pipeline selects the overall best by validation PR-AUC ─► models/best_model.joblib
```

See [mlflow.md](./mlflow.md) for how all of this is tracked.
