"""End-to-end training pipeline.

Steps:
  1. Load + validate + split (stratified 70/15/15).
  2. Train the four baseline models; tune threshold on validation; evaluate on test.
  3. Optuna-optimize each family (maximize validation PR-AUC); evaluate the best.
  4. Track everything in MLflow (baseline runs, optimized runs, nested trial runs).
  5. Select the best model by validation PR-AUC, save it, and write all reports +
     a single machine-readable dashboard_data.json used by the API and front-end.

Run:  python src/run_pipeline.py        (env N_TRIALS controls Optuna trials)
"""
from __future__ import annotations

import json
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from config import (
    ALL_MODELS, BEST_MODEL_META, BEST_MODEL_PATH, COMPARISON_CSV,
    DASHBOARD_JSON, DATA_PATH, FIGURES_DIR, FINAL_REPORT_MD, IMBALANCE_WEIGHT,
    N_TRIALS, NEGATIVE_LABEL, POSITIVE_LABEL, THRESHOLD_CSV, ensure_dirs,
)
from data import prepare_dataset
from evaluate import (
    compute_metrics, plot_confusion_matrix, plot_feature_importance,
    plot_pr_curve, plot_roc_curve,
)
from explainability import grouped_importances
from mlflow_tracking import init_mlflow, log_figure, log_metrics_prefixed
from model_factory import IMBALANCE_LABEL, fit_model
from optimize_optuna import optimize_model
from threshold_tuning import best_threshold, threshold_sweep
from train_baselines import BASELINE_PARAMS, build_baseline

warnings.filterwarnings("ignore")

def df_to_markdown(df: pd.DataFrame) -> str:
    """Minimal Markdown table renderer (avoids the optional `tabulate` dep)."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |"
            for row in df.itertuples(index=False, name=None)]
    return "\n".join([header, sep, *body])


GLOSSARY = {
    "pr_auc": "PR-AUC: how well the model separates the minority 'Declined' class across all thresholds. Best single number for imbalanced problems.",
    "roc_auc": "ROC-AUC: overall ranking quality across thresholds; can look optimistic when one class is rare.",
    "precision_declined": "Precision (Declined): among claims predicted Declined, how many were actually Declined? High precision = few false alarms.",
    "recall_declined": "Recall (Declined): among truly Declined claims, how many did the model catch? High recall = few missed declines.",
    "f1_declined": "F1 (Declined): harmonic mean of precision and recall for Declined; balances false alarms and misses.",
    "balanced_accuracy": "Balanced Accuracy: average of recall on each class; fair to both classes despite imbalance.",
    "accuracy": "Accuracy: share of all predictions correct. Misleading here -- always predicting 'Completed' already scores ~84%.",
    "confusion_matrix": "Confusion Matrix: counts of TN / FP / FN / TP at the chosen threshold.",
}


def evaluate_model(model, ds, stage, model_type, imbalance_label,
                   optuna_info=None, log_mlflow=True, want_curves=False):
    """Threshold-tune on validation, evaluate on test, plot, optionally log to MLflow."""
    # Probabilities.
    val_proba = model.predict_proba(ds.X_val)
    test_proba = model.predict_proba(ds.X_test)

    # Threshold tuning on validation (maximize F1 for Declined).
    sweep = threshold_sweep(ds.y_val, val_proba)
    thr = best_threshold(sweep, metric="f1_declined")
    model.threshold = thr

    val_metrics_default = compute_metrics(ds.y_val, val_proba, 0.5)
    val_metrics = compute_metrics(ds.y_val, val_proba, thr)
    test_metrics = compute_metrics(ds.y_test, test_proba, thr)

    # Figures (test-set based, at the chosen threshold).
    tag = f"{model_type}_{stage}"
    figs = {}
    y_test_pred = (test_proba >= thr).astype(int)
    figs["confusion_matrix"] = plot_confusion_matrix(
        ds.y_test, y_test_pred, FIGURES_DIR / f"{tag}_confusion.png",
        title=f"{model_type} ({stage}) — Test Confusion @ thr={thr:.2f}")
    figs["pr_curve"] = plot_pr_curve(
        ds.y_test, test_proba, FIGURES_DIR / f"{tag}_pr.png",
        title=f"{model_type} ({stage}) — Test PR Curve")
    figs["roc_curve"] = plot_roc_curve(
        ds.y_test, test_proba, FIGURES_DIR / f"{tag}_roc.png",
        title=f"{model_type} ({stage}) — Test ROC Curve")
    grouped = grouped_importances(model)
    if grouped:
        names = list(grouped.keys())
        vals = list(grouped.values())
        figs["feature_importance"] = plot_feature_importance(
            names, vals, FIGURES_DIR / f"{tag}_importance.png",
            title=f"{model_type} ({stage}) — Feature Importance")

    # Curve data (only kept for the best model to bound JSON size).
    curves = {}
    if want_curves:
        prec, rec, _ = precision_recall_curve(ds.y_test, test_proba)
        fpr, tpr, _ = roc_curve(ds.y_test, test_proba)
        curves = {
            "pr": {"recall": rec.round(4).tolist(), "precision": prec.round(4).tolist()},
            "roc": {"fpr": fpr.round(4).tolist(), "tpr": tpr.round(4).tolist()},
        }

    if log_mlflow:
        with mlflow.start_run(run_name=tag):
            mlflow.set_tag("model", model_type)
            mlflow.set_tag("stage", stage)
            mlflow.set_tag("imbalance_strategy", imbalance_label)
            mlflow.set_tag("preprocessing",
                           "native categorical" if model_type == "CatBoost"
                           else "median impute + one-hot")
            mlflow.log_params({f"hp_{k}": v for k, v in model.params.items()})
            mlflow.log_param("threshold", thr)
            mlflow.log_param("imbalance_strategy", imbalance_label)
            mlflow.log_metrics({
                f"class_count_{NEGATIVE_LABEL}": ds.class_counts[NEGATIVE_LABEL],
                f"class_count_{POSITIVE_LABEL}": ds.class_counts[POSITIVE_LABEL],
            })
            log_metrics_prefixed(val_metrics, "val")
            log_metrics_prefixed(test_metrics, "test")
            for f in figs.values():
                log_figure(f)
            if optuna_info:
                mlflow.log_metric("optuna_best_val_pr_auc", optuna_info["best_value"])
            # Model artifact.
            with tempfile.TemporaryDirectory() as td:
                mp = Path(td) / "claim_model.joblib"
                joblib.dump(model, mp)
                mlflow.log_artifact(str(mp), artifact_path="model")

    row = {
        "model": model_type,
        "stage": stage,
        "imbalance_strategy": imbalance_label,
        "params": model.params,
        "threshold": round(thr, 4),
        "val_pr_auc": round(val_metrics["pr_auc"], 4),
        "test_pr_auc": round(test_metrics["pr_auc"], 4),
        "test_roc_auc": round(test_metrics["roc_auc"], 4),
        "test_f1_declined": round(test_metrics["f1_declined"], 4),
        "test_recall_declined": round(test_metrics["recall_declined"], 4),
        "test_precision_declined": round(test_metrics["precision_declined"], 4),
        "test_balanced_accuracy": round(test_metrics["balanced_accuracy"], 4),
        "test_accuracy": round(test_metrics["accuracy"], 4),
        "false_positives": test_metrics["false_positives"],
        "false_negatives": test_metrics["false_negatives"],
    }
    extras = {
        "val_metrics_default": val_metrics_default,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "sweep": sweep,
        "figures": {k: str(Path(v).relative_to(FIGURES_DIR.parent)) for k, v in figs.items()},
        "feature_importance": [{"feature": k, "importance": round(v, 6)}
                               for k, v in grouped.items()][:25],
        "curves": curves,
        "confusion_matrix_test": test_metrics["confusion_matrix"],
    }
    return row, extras, model


def explain_selection(best_row, rows, ds) -> str:
    alt = [r for r in rows if not (r["model"] == best_row["model"] and r["stage"] == best_row["stage"])]
    best_alt = max(alt, key=lambda r: r["val_pr_auc"]) if alt else None
    base_rate = ds.class_counts[POSITIVE_LABEL] / sum(ds.class_counts.values())
    lift = best_row["test_pr_auc"] / base_rate if base_rate else float("nan")
    lines = [
        f"**{best_row['model']} ({best_row['stage']})** was selected as the best model.",
        "",
        f"- It achieved the highest **validation PR-AUC = {best_row['val_pr_auc']}**, the metric used for selection. "
        f"On the held-out test set it scores **PR-AUC = {best_row['test_pr_auc']}** "
        f"(~{lift:.1f}x the {base_rate:.2%} base rate of Declined), ROC-AUC = {best_row['test_roc_auc']}.",
        f"- For the minority **Declined** class it reaches recall = {best_row['test_recall_declined']}, "
        f"precision = {best_row['test_precision_declined']}, F1 = {best_row['test_f1_declined']} "
        f"at the tuned threshold {best_row['threshold']}.",
        f"- Balanced Accuracy = {best_row['test_balanced_accuracy']} (treats both classes fairly), "
        f"while raw Accuracy = {best_row['test_accuracy']}.",
    ]
    if best_alt:
        lines.append(
            f"- It edges out the next-best alternative, **{best_alt['model']} ({best_alt['stage']})** "
            f"(validation PR-AUC = {best_alt['val_pr_auc']}, test PR-AUC = {best_alt['test_pr_auc']}).")
    lines += [
        "",
        "**Why not Accuracy?** Declined claims are only ~16% of the data, so a model that blindly "
        "predicts 'Completed' would already score ~84% accuracy while catching zero declines. Accuracy "
        "rewards that useless behaviour; PR-AUC, Recall and F1 for Declined do not.",
        "",
        "**Why PR-AUC?** It summarises how well the model ranks the rare Declined class across every "
        "possible threshold, focusing on the precision/recall trade-off that matters operationally.",
        "",
        f"**Precision vs Recall trade-off & threshold.** The decision threshold ({best_row['threshold']}) was "
        "tuned on the validation set to maximise F1 for Declined. Lowering it catches more declines (higher "
        "recall) at the cost of more false alarms (lower precision); raising it does the opposite. The "
        "business can move this slider depending on the cost of a missed decline vs a false alarm.",
    ]
    return "\n".join(lines)


def main():
    ensure_dirs()
    init_mlflow()
    ds, report = prepare_dataset(DATA_PATH)

    # Per-feature metadata for the prediction demo (defaults + categorical options),
    # computed from the training split only.
    feature_meta = {}
    for c in ds.num_cols:
        feature_meta[c] = {"type": "numeric",
                           "default": float(np.nanmedian(pd.to_numeric(ds.X_train[c], errors="coerce")))}
    for c in ds.cat_cols:
        vals = ds.X_train[c].astype(str)
        options = [str(v) for v in vals.value_counts().index.tolist()[:15]]
        feature_meta[c] = {"type": "categorical", "default": options[0] if options else "Unknown",
                           "options": options}

    rows, extras_by_key, models_by_key, optuna_by_model = [], {}, {}, {}

    # --- 1. Baselines ----------------------------------------------------
    print("\n" + "#" * 70 + "\n# BASELINE MODELS\n" + "#" * 70)
    for mt in ALL_MODELS:
        print(f"[baseline] {mt} ...", flush=True)
        model = build_baseline(mt, ds)
        row, extras, model = evaluate_model(
            model, ds, "baseline", mt, IMBALANCE_LABEL[mt], want_curves=False)
        rows.append(row)
        key = (mt, "baseline")
        extras_by_key[key] = extras
        models_by_key[key] = model
        print(f"           val PR-AUC={row['val_pr_auc']} test PR-AUC={row['test_pr_auc']} "
              f"thr={row['threshold']} F1d={row['test_f1_declined']}")

    # --- 2. Optuna optimization -----------------------------------------
    print("\n" + "#" * 70 + f"\n# OPTUNA OPTIMIZATION ({N_TRIALS} trials/model)\n" + "#" * 70)
    for mt in ALL_MODELS:
        print(f"[optuna] {mt} ...", flush=True)
        info = optimize_model(mt, ds, N_TRIALS, log_mlflow=True)
        optuna_by_model[mt] = info
        model = fit_model(mt, info["best_params"], info["best_imbalance"], ds)
        imb_label = info["best_imbalance"] if isinstance(info["best_imbalance"], str) else str(info["best_imbalance"])
        row, extras, model = evaluate_model(
            model, ds, "optimized", mt, imb_label, optuna_info=info, want_curves=True)
        rows.append(row)
        key = (mt, "optimized")
        extras_by_key[key] = extras
        models_by_key[key] = model
        print(f"          best val PR-AUC(search)={info['best_value']:.4f} "
              f"| eval val PR-AUC={row['val_pr_auc']} test PR-AUC={row['test_pr_auc']} "
              f"thr={row['threshold']} F1d={row['test_f1_declined']}")

    # --- 3. Select best by validation PR-AUC (tie-break: test F1) --------
    best_row = max(rows, key=lambda r: (r["val_pr_auc"], r["test_f1_declined"]))
    best_key = (best_row["model"], best_row["stage"])
    best_model = models_by_key[best_key]
    best_extras = extras_by_key[best_key]
    for r in rows:
        r["is_best"] = (r["model"] == best_row["model"] and r["stage"] == best_row["stage"])

    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best_row['model']} ({best_row['stage']}) "
          f"val PR-AUC={best_row['val_pr_auc']} test PR-AUC={best_row['test_pr_auc']}")
    print("=" * 70)

    # --- 4. Save best model + metadata -----------------------------------
    joblib.dump(best_model, BEST_MODEL_PATH)
    trained_at = datetime.now(timezone.utc)
    meta = {
        "model_version": f"{best_model.model_type.lower()}-{best_row['stage']}-"
                         f"{trained_at.strftime('%Y%m%d')}",
        "trained_at": trained_at.isoformat(),
        "model_type": best_model.model_type,
        "stage": best_row["stage"],
        "imbalance_strategy": best_row["imbalance_strategy"],
        "params": best_model.params,
        "threshold": best_model.threshold,
        "feature_cols": best_model.feature_cols,
        "num_cols": best_model.num_cols,
        "cat_cols": best_model.cat_cols,
        "feature_importance": best_extras["feature_importance"],
        "positive_class": POSITIVE_LABEL,
        "negative_class": NEGATIVE_LABEL,
        "test_metrics": best_extras["test_metrics"],
        "optuna_trials": N_TRIALS,
    }
    BEST_MODEL_META.write_text(json.dumps(meta, indent=2))

    # --- 5. Reports ------------------------------------------------------
    comp_cols = ["model", "stage", "imbalance_strategy", "threshold", "val_pr_auc",
                 "test_pr_auc", "test_roc_auc", "test_f1_declined", "test_recall_declined",
                 "test_precision_declined", "test_balanced_accuracy", "test_accuracy",
                 "false_positives", "false_negatives", "is_best"]
    comp_df = pd.DataFrame([{c: r.get(c) for c in comp_cols} for r in rows])
    comp_df = comp_df.sort_values("val_pr_auc", ascending=False).reset_index(drop=True)
    comp_df.to_csv(COMPARISON_CSV, index=False)

    best_extras["sweep"].to_csv(THRESHOLD_CSV, index=False)

    explanation = explain_selection(best_row, rows, ds)

    # final_model_report.md
    md = ["# Final Model Report — Claim Approval (`status`)\n",
          f"_Positive class: **{POSITIVE_LABEL}** (1). Negative class: **{NEGATIVE_LABEL}** (0)._\n",
          "## Model Comparison\n",
          df_to_markdown(comp_df.drop(columns=["is_best"])),
          "\n## Best Model Selection\n", explanation,
          "\n## Best Hyperparameters\n",
          "```json\n" + json.dumps(best_model.params, indent=2) + "\n```\n",
          f"\n## Key Risk / Limitation\n",
          "- The available features give only a moderate separating signal for declines "
          f"(test PR-AUC = {best_row['test_pr_auc']}); some genuine declines are still missed "
          f"(false negatives = {best_row['false_negatives']} on the test set). "
          "Predictions should support, not replace, human adjudication.\n",
          "\n## Final Recommendation\n",
          "```text",
          f"Best Model: {best_row['model']} ({best_row['stage']})",
          f"Best Imbalance Strategy: {best_row['imbalance_strategy']}",
          f"Best Hyperparameters: {json.dumps(best_model.params)}",
          f"Best Threshold: {best_model.threshold}",
          f"Main Reason for Selection: highest validation PR-AUC ({best_row['val_pr_auc']}); "
          f"strong test PR-AUC ({best_row['test_pr_auc']}) and Declined recall "
          f"({best_row['test_recall_declined']}) with balanced accuracy "
          f"({best_row['test_balanced_accuracy']}).",
          f"Key Risk or Limitation: moderate signal — {best_row['false_negatives']} declines missed "
          "on test; use as decision support with human review.",
          "```\n"]
    FINAL_REPORT_MD.write_text("\n".join(md))

    # --- 6. dashboard_data.json (consumed by API + front-end) ------------
    dashboard = {
        "generated_with": {"n_trials": N_TRIALS, "data": str(DATA_PATH.name)},
        "overview": {
            "dataset_name": DATA_PATH.name,
            "n_rows": report["shape"][0],
            "n_features": len(ds.feature_cols),
            "n_numeric": len(ds.num_cols),
            "n_categorical": len(ds.cat_cols),
            "numeric_features": ds.num_cols,
            "categorical_features": ds.cat_cols,
            "target_distribution": report["target_distribution"],
            "class_counts": ds.class_counts,
            "imbalance_ratio": report["imbalance_ratio"],
            "imbalance_weight": IMBALANCE_WEIGHT,
            "split_sizes": report["split_sizes"],
            "excluded_columns": report["excluded_columns"],
            "feature_meta": feature_meta,
        },
        "leaderboard": [
            {**{k: r.get(k) for k in comp_cols}, "params": r["params"]} for r in
            sorted(rows, key=lambda r: r["val_pr_auc"], reverse=True)
        ],
        "best_model": {
            "model": best_row["model"],
            "stage": best_row["stage"],
            "imbalance_strategy": best_row["imbalance_strategy"],
            "params": best_model.params,
            "threshold": best_model.threshold,
            "test_metrics": best_extras["test_metrics"],
            "val_metrics": best_extras["val_metrics"],
            "feature_importance": best_extras["feature_importance"],
            "confusion_matrix_test": best_extras["confusion_matrix_test"],
            "curves": best_extras["curves"],
            "figures": best_extras["figures"],
            "explanation": explanation,
        },
        "threshold_analysis": best_extras["sweep"].round(4).to_dict(orient="records"),
        "optuna": {
            mt: {
                "best_value": info["best_value"],
                "best_params": info["best_params"],
                "param_importances": info["param_importances"],
                "history": [{"trial": h["trial"], "value": h["value"]} for h in info["history"]],
                "n_trials": info["n_trials"],
            } for mt, info in optuna_by_model.items()
        },
        "metric_glossary": GLOSSARY,
        "final_recommendation": {
            "best_model": f"{best_row['model']} ({best_row['stage']})",
            "imbalance_strategy": best_row["imbalance_strategy"],
            "hyperparameters": best_model.params,
            "threshold": best_model.threshold,
            "reason": f"Highest validation PR-AUC ({best_row['val_pr_auc']}); strong test PR-AUC "
                      f"({best_row['test_pr_auc']}) and Declined recall ({best_row['test_recall_declined']}).",
            "risk": f"Moderate signal; {best_row['false_negatives']} declines missed on test — use as decision support.",
        },
    }
    DASHBOARD_JSON.write_text(json.dumps(dashboard, indent=2, default=float))

    # Copy dashboard JSON into the model-dashboard front-end public dir if it exists.
    fe_public = Path(__file__).resolve().parents[1] / "model-dashboard" / "public"
    if fe_public.exists():
        (fe_public / "dashboard_data.json").write_text(json.dumps(dashboard, default=float))

    print("\nArtifacts written:")
    for p in [BEST_MODEL_PATH, BEST_MODEL_META, COMPARISON_CSV, THRESHOLD_CSV,
              FINAL_REPORT_MD, DASHBOARD_JSON]:
        print(f"  - {p}")
    print("\nFINAL RECOMMENDATION:")
    print(json.dumps(dashboard["final_recommendation"], indent=2, default=float))


if __name__ == "__main__":
    main()
