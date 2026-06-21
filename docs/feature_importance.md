# How feature importance is calculated

**Question it answers:** *which input features did the final fitted model rely on most?* This drives
the "Feature Importance" chart on the **Final Model** dashboard page, the `*_importance.png` figures
in `reports/figures/`, and the `/feature-importance` API endpoint.

This is about **features** (e.g. `rrp`, `retailerName`) — distinct from
[hyperparameter importance](./hyperparameter_importance.md), which is about the search.

## Where it happens

Two steps:

1. **`src/model_factory.py` → `ClaimModel.feature_importance()`** — pulls the raw importance from the
   fitted estimator, aligned with the feature names the estimator actually saw.
2. **`src/ml/explainability.py` → `grouped_importances()`** — rolls one-hot dummy columns back up to
   their original feature, sorts descending, drops zeros.

The grouped result is stored in `dashboard_data.json` at `best_model.feature_importance` (top 25),
plotted by `ml.evaluate.plot_feature_importance`, and shown in `model-dashboard/src/pages/FinalModel.jsx`.

## The raw importance — method depends on the model family

Each library exposes its own built-in importance; we use each library's value as-is. They are
**model-internal** importances (how much the model *used* a feature), **not** SHAP or permutation
importance.

| Family | Source (in code) | Importance type | Normalization | Feature space |
|---|---|---|---|---|
| **RandomForest** | `estimator.feature_importances_` | **MDI / Gini** — mean decrease in impurity across all splits on the feature | sums to 1.0 | one-hot + numeric |
| **XGBoost** | `estimator.feature_importances_` | **Gain** — average gain (loss reduction) of splits using the feature (the sklearn-wrapper default) | sums to 1.0 | one-hot + numeric |
| **LightGBM** | `estimator.feature_importances_` | **Split count** — number of times the feature is used to split (`importance_type='split'`, the default) | raw counts (not normalized) | one-hot + numeric |
| **CatBoost** | `estimator.get_feature_importance()` | **PredictionValuesChange** — average change in the prediction when the feature's value changes (CatBoost default) | sums to ~100 | **original** features (native categoricals, no one-hot) |

### Why the feature space differs
- RandomForest / XGBoost / LightGBM are fed the **one-hot encoded** matrix from the
  `ColumnTransformer`, so their raw importances are per *encoded* column. We align them with
  `preprocessor.get_feature_names_out()` (e.g. `retailerName_EVOLLIS`, `rrp`).
- CatBoost uses **native categorical handling** (no one-hot), so its importances are already per
  *original* feature (`retailerName`, `rrp`, …) and align with `feature_cols`.

> The current best model is **CatBoost**, so its feature importances are *PredictionValuesChange*
> over the original features.

## Grouping one-hot dummies back to the original feature

`explainability.grouped_importances()` makes the chart readable for tree models: instead of 30 tiny
`retailerName_*` bars, it **sums** the importance of all dummy columns that belong to a categorical
column back into that single feature.

```
retailerName_EVOLLIS + retailerName_WUAWEI eStore + … ─► retailerName
rrp (numeric)                                         ─► rrp   (unchanged)
```

Numeric features pass through unchanged. The result is sorted descending and zero-importance features
are dropped. For CatBoost this step is effectively a pass-through (already per original feature).

## Important caveats

- **Model-internal, not causal.** These measure how much the model *used* a feature for its
  predictions, not the feature's true causal effect on `status`.
- **Metrics are not directly comparable across families.** MDI vs gain vs split-count vs
  PredictionValuesChange are different scales; use them to rank reliance *within* a model, not to
  compare absolute numbers between models.
- **Known biases.** Impurity/MDI (RandomForest) and split-count (LightGBM) tend to inflate
  high-cardinality or continuous features; that's a property of the metric, mitigated partly by the
  one-hot grouping above.
- **Computed on the fitted best model** after training — it is a property of that model, independent
  of the threshold or the test set.
- For a model-agnostic, leakage-robust view you would add SHAP or permutation importance; that is a
  possible extension, not what this repo currently computes.

## Related

- [hyperparameter_importance.md](./hyperparameter_importance.md) — importance over the *search*, not
  the features.
- [mlflow.md](./mlflow.md) — where the importance figures are logged as run artifacts.
