# Final Model Report — Claim Approval (`status`)

_Positive class: **Declined** (1). Negative class: **Completed** (0)._

## Model Comparison

| model | stage | imbalance_strategy | threshold | val_pr_auc | test_pr_auc | test_roc_auc | test_f1_declined | test_recall_declined | test_precision_declined | test_balanced_accuracy | test_accuracy | false_positives | false_negatives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CatBoost | optimized | class_weights | 0.44 | 0.3056 | 0.3027 | 0.6791 | 0.3269 | 0.75 | 0.209 | 0.6099 | 0.5139 | 193 | 17 |
| XGBoost | optimized | scale_pos_weight=1.65 | 0.19 | 0.2953 | 0.2878 | 0.6514 | 0.322 | 0.7647 | 0.2039 | 0.6035 | 0.4931 | 203 | 16 |
| LightGBM | optimized | scale_pos_weight=1.05 | 0.15 | 0.2849 | 0.3114 | 0.6582 | 0.3228 | 0.75 | 0.2056 | 0.6044 | 0.5046 | 197 | 17 |
| RandomForest | optimized | custom | 0.51 | 0.272 | 0.3062 | 0.6648 | 0.3543 | 0.6618 | 0.2419 | 0.6372 | 0.6204 | 141 | 23 |
| XGBoost | baseline | scale_pos_weight=5.36 | 0.47 | 0.2526 | 0.3047 | 0.6945 | 0.3516 | 0.7059 | 0.2341 | 0.6373 | 0.5903 | 157 | 20 |
| CatBoost | baseline | class_weights={Completed:1, Declined:5.36} | 0.46 | 0.2466 | 0.2985 | 0.6607 | 0.3273 | 0.6618 | 0.2174 | 0.6084 | 0.5718 | 162 | 23 |
| LightGBM | baseline | scale_pos_weight=5.36 | 0.19 | 0.244 | 0.2313 | 0.6346 | 0.3333 | 0.7059 | 0.2182 | 0.6167 | 0.5556 | 172 | 20 |
| RandomForest | baseline | class_weight='balanced' | 0.47 | 0.2133 | 0.3264 | 0.6904 | 0.3777 | 0.6471 | 0.2667 | 0.6573 | 0.6644 | 121 | 24 |

## Best Model Selection

**CatBoost (optimized)** was selected as the best model.

- It achieved the highest **validation PR-AUC = 0.3056**, the metric used for selection. On the held-out test set it scores **PR-AUC = 0.3027** (~1.9x the 15.73% base rate of Declined), ROC-AUC = 0.6791.
- For the minority **Declined** class it reaches recall = 0.75, precision = 0.209, F1 = 0.3269 at the tuned threshold 0.44.
- Balanced Accuracy = 0.6099 (treats both classes fairly), while raw Accuracy = 0.5139.
- It edges out the next-best alternative, **XGBoost (optimized)** (validation PR-AUC = 0.2953, test PR-AUC = 0.2878).

**Why not Accuracy?** Declined claims are only ~16% of the data, so a model that blindly predicts 'Completed' would already score ~84% accuracy while catching zero declines. Accuracy rewards that useless behaviour; PR-AUC, Recall and F1 for Declined do not.

**Why PR-AUC?** It summarises how well the model ranks the rare Declined class across every possible threshold, focusing on the precision/recall trade-off that matters operationally.

**Precision vs Recall trade-off & threshold.** The decision threshold (0.44) was tuned on the validation set to maximise F1 for Declined. Lowering it catches more declines (higher recall) at the cost of more false alarms (lower precision); raising it does the opposite. The business can move this slider depending on the cost of a missed decline vs a false alarm.

## Best Hyperparameters

```json
{
  "iterations": 600,
  "depth": 4,
  "learning_rate": 0.1530883741573138,
  "l2_leaf_reg": 1.6709557931179373,
  "bagging_temperature": 0.9868869366005173,
  "random_strength": 1.5444895385933148,
  "border_count": 76
}
```


## Key Risk / Limitation

- The available features give only a moderate separating signal for declines (test PR-AUC = 0.3027); some genuine declines are still missed (false negatives = 17 on the test set). Predictions should support, not replace, human adjudication.


## Final Recommendation

```text
Best Model: CatBoost (optimized)
Best Imbalance Strategy: class_weights
Best Hyperparameters: {"iterations": 600, "depth": 4, "learning_rate": 0.1530883741573138, "l2_leaf_reg": 1.6709557931179373, "bagging_temperature": 0.9868869366005173, "random_strength": 1.5444895385933148, "border_count": 76}
Best Threshold: 0.44
Main Reason for Selection: highest validation PR-AUC (0.3056); strong test PR-AUC (0.3027) and Declined recall (0.75) with balanced accuracy (0.6099).
Key Risk or Limitation: moderate signal — 17 declines missed on test; use as decision support with human review.
```
