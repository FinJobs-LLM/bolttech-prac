# How hyperparameter importance is calculated

**Question it answers:** *for a given model family, which hyperparameters most influenced the search
objective (validation PR-AUC)?* This is what drives the "Parameter importance" bar chart on the
**MLflow / Optuna** dashboard page.

It is computed **per model family**, from that family's Optuna study — i.e. from the 50 trials and
their validation PR-AUC scores, **not** from the final fitted model.

## Where it happens

`src/optimize_optuna.py` (in `optimize_model`):

```python
importances = {k: float(v) for k, v in
               optuna.importance.get_param_importances(study).items()}
```

The result is stored in `dashboard_data.json` at `optuna.<Model>.param_importances` and rendered by
`frontend/src/pages/Tracking.jsx`. The call is wrapped in `try/except` → `{}` when importances can't
be computed (e.g. too few completed trials, or no variation in a parameter).

## The method: fANOVA

`optuna.importance.get_param_importances` uses Optuna's **default evaluator,
`FanovaImportanceEvaluator`** (functional ANOVA). The procedure:

1. Take every completed trial in the study as a data point: *(sampled hyperparameters → objective
   value)*, where the objective is **validation PR-AUC** (see [optuna.md](./optuna.md)).
2. Fit a **random-forest surrogate model** that predicts the objective from the hyperparameters.
3. Decompose the **variance** of the objective into the contribution of each hyperparameter (and
   their interactions) using functional ANOVA on that surrogate.
4. Report each hyperparameter's share of the explained variance.

**Interpretation:** a high value means *"varying this hyperparameter explained a large part of the
spread in validation PR-AUC across the search"* — i.e. the score was sensitive to it. The values are
**normalized to sum to 1.0** within a study.

## What goes in, what comes out

- **Input:** the trials of one study (`study.trials`) — their sampled params and PR-AUC values.
  Categorical hyperparameters (e.g. CatBoost's `class_weights` vs `auto_class_weights`,
  RandomForest's `class_weight`) are supported.
- **Output:** `{hyperparameter: importance}` summing to 1.0, e.g.
  `{"learning_rate": 0.41, "depth": 0.22, ...}`.

## Important caveats

- **Sensitivity, not optimality.** A high importance means the objective was sensitive to that
  hyperparameter; it does **not** say which *value* is best (that's `best_params`).
- **Relative to the search space.** Importance depends on the ranges defined in
  `suggest_params`. A hyperparameter given a narrow range will look unimportant simply because it
  barely varied.
- **Per study / not cross-comparable.** Each family has its own study, so importances are comparable
  *within* a family, not across families.
- **Needs enough varied trials.** With very few trials (e.g. a smoke run) the estimate is noisy or
  empty — hence the `try/except`.
- It reflects influence on the **validation** objective, which is itself a finite, ~432-row sample —
  see the over-fitting discussion in `reports/final_model_report.md` / the project history.

## Related

- [optuna.md](./optuna.md) — the search that produces the studies.
- [feature_importance.md](./feature_importance.md) — a *different* quantity: how the final fitted
  model relies on input **features** (not hyperparameters).
