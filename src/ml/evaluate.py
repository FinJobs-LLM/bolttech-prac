"""Metrics and plotting for imbalanced binary classification.

`Declined` is the positive class (label 1). Threshold-independent metrics
(PR-AUC, ROC-AUC) use predicted probabilities; the rest use a decision
threshold applied to the probability of the positive class.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """All tracked metrics for a given probability vector and threshold."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    # Threshold-independent (use probabilities).
    pr_auc = float(average_precision_score(y_true, y_proba))
    try:
        roc_auc = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        roc_auc = float("nan")

    # Positive-class (Declined) precision/recall/F1 at this threshold.
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average=None, zero_division=0
    )
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    acc = float((y_pred == y_true).mean())

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    return {
        "threshold": float(threshold),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "f1_declined": float(f1[0]),
        "recall_declined": float(rec[0]),
        "precision_declined": float(prec[0]),
        "balanced_accuracy": bal_acc,
        "accuracy": acc,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


# --- Plots ---------------------------------------------------------------
def _save(fig, path: Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_confusion_matrix(y_true, y_pred, path: Path, title: str = "Confusion Matrix") -> str:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Completed (0)", "Declined (1)"]
    ax.set_xticks([0, 1], labels, rotation=15)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, path)


def plot_pr_curve(y_true, y_proba, path: Path, title: str = "Precision-Recall Curve") -> str:
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    baseline = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(rec, prec, color="#2563eb", lw=2, label=f"PR-AUC = {ap:.3f}")
    ax.axhline(baseline, color="gray", ls="--", lw=1, label=f"baseline = {baseline:.3f}")
    ax.set_xlabel("Recall (Declined)")
    ax.set_ylabel("Precision (Declined)")
    ax.set_title(title)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_roc_curve(y_true, y_proba, path: Path, title: str = "ROC Curve") -> str:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(fpr, tpr, color="#16a34a", lw=2, label=f"ROC-AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_feature_importance(names, importances, path: Path, top_n: int = 20,
                            title: str = "Feature Importance") -> str:
    names = np.asarray(names)
    importances = np.asarray(importances, dtype=float)
    order = np.argsort(importances)[::-1][:top_n]
    names, importances = names[order][::-1], importances[order][::-1]
    fig, ax = plt.subplots(figsize=(6.4, max(3.5, 0.32 * len(names))))
    ax.barh(range(len(names)), importances, color="#7c3aed")
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="x")
    return _save(fig, path)
