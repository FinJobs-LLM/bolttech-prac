"""Decision-threshold tuning on the validation set.

We never trust the default 0.5 cut-off. We sweep thresholds in [0.05, 0.95]
and, by default, pick the one that maximises F1 for the Declined class -- a
balance between catching declined claims (recall) and not crying wolf
(precision). The chosen threshold is then frozen and reused on the test set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

from config import THRESHOLD_MAX, THRESHOLD_MIN


def threshold_sweep(y_true, y_proba, step: float = 0.01) -> pd.DataFrame:
    """Return a per-threshold table of Declined precision/recall/F1, balanced acc, FP, FN."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    thresholds = np.round(np.arange(THRESHOLD_MIN, THRESHOLD_MAX + 1e-9, step), 4)
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[1], average=None, zero_division=0
        )
        bal = balanced_accuracy_score(y_true, y_pred)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        rows.append({
            "threshold": float(t),
            "precision_declined": float(prec[0]),
            "recall_declined": float(rec[0]),
            "f1_declined": float(f1[0]),
            "balanced_accuracy": float(bal),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    return pd.DataFrame(rows)


def best_threshold(sweep: pd.DataFrame, metric: str = "f1_declined") -> float:
    """Threshold maximising the chosen metric (ties broken by higher recall)."""
    best = sweep.sort_values([metric, "recall_declined"], ascending=False).iloc[0]
    return float(best["threshold"])
