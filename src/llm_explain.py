"""LLM-generated, plain-English explanation of the serving model.

Uses LangChain + OpenAI (gpt-4o-mini) to turn the model's metadata and feature
importances into a short, business-friendly narrative. The call runs server-side
so the OpenAI API key never reaches the browser.

Requires the OPENAI_API_KEY environment variable. If it is not set, the API
endpoints surface a clear 503 instead of failing opaquely.
"""
from __future__ import annotations

import os

from prompts.adjuster import ADJUSTER_HUMAN_TEMPLATE, ADJUSTER_SYSTEM_PROMPT
from prompts.customer import CUSTOMER_HUMAN_TEMPLATE, CUSTOMER_SYSTEM_PROMPT
from prompts.model_explanation import HUMAN_TEMPLATE, SYSTEM_PROMPT

MODEL_NAME = "gpt-4o-mini"


def _importance_metric(model_name: str) -> str:
    """Human description of the importance metric used by the served model type."""
    name = (model_name or "").lower()
    if "catboost" in name:
        return "CatBoost PredictionValuesChange — average change in the prediction when a feature's value changes (scores sum to ~100)"
    if "randomforest" in name or "random forest" in name:
        return "Random Forest mean decrease in impurity / Gini importance (scores sum to ~1)"
    if "xgboost" in name:
        return "XGBoost gain — average loss reduction from splits using the feature (scores sum to ~1)"
    if "lightgbm" in name:
        return "LightGBM split count — how often the feature is used to split (raw counts)"
    return "model-internal feature importance (higher = the model relies on it more)"


def explanation_available() -> bool:
    """True if an OpenAI API key is configured."""
    return bool(os.environ.get("OPENAI_API_KEY"))


def _fmt_metrics(m: dict) -> str:
    keys = [
        ("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("f1_declined", "F1 (Declined)"),
        ("recall_declined", "Recall (Declined)"), ("precision_declined", "Precision (Declined)"),
        ("balanced_accuracy", "Balanced accuracy"), ("accuracy", "Accuracy"),
    ]
    lines = []
    for k, label in keys:
        if k in m and m[k] is not None:
            lines.append(f"- {label}: {float(m[k]):.3f}")
    if "false_negatives" in m and "false_positives" in m:
        lines.append(f"- False negatives: {m['false_negatives']} | False positives: {m['false_positives']}")
    return "\n".join(lines) or "- (none provided)"


def _fmt_features(items: list, top_n: int = 15) -> str:
    rows = sorted(items, key=lambda x: x.get("importance", 0), reverse=True)[:top_n]
    return "\n".join(
        f"{i}. {r['feature']}: {float(r['importance']):.2f}" for i, r in enumerate(rows, 1)
    ) or "- (none)"


def generate_model_explanation(model_info: dict, feature_importance: list,
                               temperature: float = 0.6) -> str:
    """Call gpt-4o-mini via LangChain and return the explanation text.

    Raises RuntimeError if OPENAI_API_KEY is missing.
    """
    if not explanation_available():
        raise RuntimeError("OPENAI_API_KEY is not set on the server.")

    # Imported lazily so the rest of the API works without langchain installed.
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=MODEL_NAME, temperature=temperature, max_tokens=750, timeout=60)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_TEMPLATE)]
    )
    model_name = model_info.get("model_name") or model_info.get("model") or "unknown"
    chain = prompt | llm
    resp = chain.invoke({
        "model_name": model_name,
        "stage": model_info.get("stage", ""),
        "model_version": model_info.get("model_version") or "n/a",
        "imbalance_strategy": model_info.get("imbalance_strategy", "n/a"),
        "threshold": model_info.get("threshold", "n/a"),
        "metrics_block": _fmt_metrics(model_info.get("test_metrics", {}) or {}),
        "importance_metric": _importance_metric(model_name),
        "features_block": _fmt_features(feature_importance or []),
    })
    return resp.content.strip()


# ---------------------------------------------------------------------------
# Per-prediction explanation for a claims adjuster (decision support only)
# Prompts: prompts/adjuster.py
# ---------------------------------------------------------------------------
def _fmt_drivers(feature_importance: list, claim_features: dict, top_n: int = 12) -> str:
    rows = sorted(feature_importance or [], key=lambda x: x.get("importance", 0), reverse=True)[:top_n]
    lines = []
    for i, r in enumerate(rows, 1):
        f = r["feature"]
        v = claim_features.get(f, None)
        v = "(blank/Unknown)" if v is None or v == "" else v
        lines.append(f"{i}. {f} (importance {float(r['importance']):.2f}) — claim value: {v}")
    return "\n".join(lines) or "- (no feature importance available)"


def generate_prediction_explanation(prediction: dict, claim_features: dict,
                                    model_info: dict, feature_importance: list,
                                    temperature: float = 0.4) -> str:
    """Adjuster-facing explanation of a SINGLE model prediction.

    The model has already made the prediction; this only explains it. Raises
    RuntimeError if OPENAI_API_KEY is missing.
    """
    if not explanation_available():
        raise RuntimeError("OPENAI_API_KEY is not set on the server.")

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=MODEL_NAME, temperature=temperature, max_tokens=750, timeout=60)
    prompt = ChatPromptTemplate.from_messages(
        [("system", ADJUSTER_SYSTEM_PROMPT), ("human", ADJUSTER_HUMAN_TEMPLATE)]
    )
    model_name = model_info.get("model_name") or model_info.get("model") or "unknown"
    chain = prompt | llm
    resp = chain.invoke({
        "predicted_class": prediction.get("predicted_class", "n/a"),
        "p_declined": f"{float(prediction.get('probability_declined', 0)):.1%}",
        "p_completed": f"{float(prediction.get('probability_completed', 0)):.1%}",
        "threshold": prediction.get("threshold_used", model_info.get("threshold", "n/a")),
        "model_name": model_name,
        "metrics_block": _fmt_metrics(model_info.get("test_metrics", {}) or {}),
        "importance_metric": _importance_metric(model_name),
        "drivers_block": _fmt_drivers(feature_importance or [], claim_features or {}),
    })
    return resp.content.strip()


# ---------------------------------------------------------------------------
# Per-prediction explanation for the CUSTOMER (plain language, no jargon)
# Prompts: prompts/customer.py
# ---------------------------------------------------------------------------
def _customer_outcome(prediction: dict) -> str:
    label = prediction.get("predicted_label")
    cls = prediction.get("predicted_class", "")
    if label == 1 or cls == "Declined":
        return ("the system leaned toward expecting that this claim might not be approved "
                "(it may be declined) — this is only a preliminary, automated result")
    return ("the system leaned toward expecting that this claim is likely to be approved / "
            "go through — this is only a preliminary, automated result")


def _fmt_customer_drivers(feature_importance: list, claim_features: dict, top_n: int = 8) -> str:
    """Ranked influential details + the claim's values, WITHOUT importance numbers."""
    rows = sorted(feature_importance or [], key=lambda x: x.get("importance", 0), reverse=True)[:top_n]
    lines = []
    for i, r in enumerate(rows, 1):
        f = r["feature"]
        v = claim_features.get(f, None)
        v = "(not provided)" if v is None or v == "" else v
        lines.append(f"{i}. {f} = {v}")
    return "\n".join(lines) or "- (no details available)"


def generate_customer_explanation(prediction: dict, claim_features: dict,
                                  model_info: dict, feature_importance: list,
                                  temperature: float = 0.5) -> str:
    """Customer-facing, plain-language explanation of a SINGLE model prediction.

    The model has already made the prediction; this only explains it in simple
    terms. Raises RuntimeError if OPENAI_API_KEY is missing.
    """
    if not explanation_available():
        raise RuntimeError("OPENAI_API_KEY is not set on the server.")

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=MODEL_NAME, temperature=temperature, max_tokens=600, timeout=60)
    prompt = ChatPromptTemplate.from_messages(
        [("system", CUSTOMER_SYSTEM_PROMPT), ("human", CUSTOMER_HUMAN_TEMPLATE)]
    )
    chain = prompt | llm
    resp = chain.invoke({
        "outcome_plain": _customer_outcome(prediction),
        "drivers_block": _fmt_customer_drivers(feature_importance or [], claim_features or {}),
    })
    return resp.content.strip()
