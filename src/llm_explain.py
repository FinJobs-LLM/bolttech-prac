"""LLM-generated, plain-English explanation of the serving model.

Uses LangChain + OpenAI (gpt-4o-mini) to turn the model's metadata and feature
importances into a short, business-friendly narrative. The call runs server-side
so the OpenAI API key never reaches the browser.

Requires the OPENAI_API_KEY environment variable. If it is not set, the API
endpoints surface a clear 503 instead of failing opaquely.
"""
from __future__ import annotations

import os

MODEL_NAME = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are an ML explainability assistant for an insurance claim-approval model. "
    "Explain in clear, business-friendly English for a semi-technical audience. "
    "The model predicts a claim's `status`; the POSITIVE class is `Declined` (a claim "
    "the model flags as likely to be declined). The data is imbalanced (~16% Declined), "
    "so PR-AUC, recall and precision for Declined matter more than raw accuracy. "
    "Always include a dedicated 'Feature importance' section that (a) briefly explains, in plain "
    "words, what the importance scores represent for this model, and (b) walks through the most "
    "influential features in rank order with a short, clearly-hedged business intuition for each, "
    "noting these are how-much-the-model-relies-on-a-feature signals, not proven causes. "
    "Rules: only use the numbers/features provided — never invent metrics or features; keep it to "
    "roughly 250-350 words; use short paragraphs and bullets with section headings; do not give "
    "financial advice; end with one sentence on a key limitation."
)

HUMAN_TEMPLATE = (
    "Explain this served model and what drives its predictions.\n\n"
    "MODEL\n"
    "- Name: {model_name} ({stage})\n"
    "- Version: {model_version}\n"
    "- Imbalance strategy: {imbalance_strategy}\n"
    "- Decision threshold: {threshold}\n\n"
    "TEST METRICS (held-out)\n{metrics_block}\n\n"
    "FEATURE IMPORTANCE — metric: {importance_metric}\n"
    "(ranked, higher = more influential)\n{features_block}\n\n"
    "Write these sections with headings:\n"
    "(1) What the model does and how to read its decisions at the threshold.\n"
    "(2) How good it is — emphasise PR-AUC / recall / precision for Declined over accuracy, and why.\n"
    "(3) Feature importance — first explain in one or two sentences what the importance metric "
    "above measures, then interpret the ranked features: cover the top 5-7 individually with a "
    "short hedged intuition for each, and briefly note any features that barely matter. Make clear "
    "these reflect what the model relies on, not proven cause and effect.\n"
    "(4) One key limitation."
)


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
# ---------------------------------------------------------------------------
ADJUSTER_SYSTEM_PROMPT = (
    "You are an assistant to a human INSURANCE CLAIMS ADJUSTER who is manually reviewing a single "
    "claim. An ML model has ALREADY produced a prediction for this claim. The POSITIVE class is "
    "`Declined`. Your ONLY job is to explain the model's output so the adjuster can do an informed "
    "manual review.\n"
    "STRICT RULES:\n"
    "- You DO NOT make, recommend, endorse, or imply a decision. Never say the claim should be "
    "approved/declined/paid/denied, and never tell the adjuster what to decide.\n"
    "- You do not override or second-guess the model; you interpret what it produced and why it may "
    "have produced it, as decision-support only.\n"
    "- Use ONLY the prediction, model metadata, feature importances, and claim values provided. "
    "Never invent feature values, metrics, or claim facts.\n"
    "- Be professional and concise (~250-350 words), in clear adjuster-friendly language. "
    "Calibrate confidence honestly using the model's precision/recall. The final decision is the "
    "adjuster's; state this once at the end."
)

ADJUSTER_HUMAN_TEMPLATE = (
    "Explain this model prediction to support manual review. Do NOT make the decision.\n\n"
    "MODEL PREDICTION FOR THIS CLAIM\n"
    "- Model output: {predicted_class}\n"
    "- P(Declined): {p_declined} | P(Completed): {p_completed}\n"
    "- Decision threshold: {threshold} (flagged Declined when P(Declined) >= threshold)\n\n"
    "MODEL ({model_name}) RELIABILITY — held-out test metrics\n{metrics_block}\n\n"
    "WHAT THE MODEL GENERALLY WEIGHS, WITH THIS CLAIM'S VALUES\n"
    "(importance metric: {importance_metric}; importances are GLOBAL, not a per-claim attribution)\n"
    "{drivers_block}\n\n"
    "Write these sections with headings:\n"
    "(1) Model prediction — restate the output and what the probability/threshold mean for this claim.\n"
    "(2) How much to trust it — interpret precision and recall for Declined in practical terms "
    "(e.g. false-positive vs false-negative risk) so the adjuster knows how cautiously to treat it.\n"
    "(3) Likely contributing factors — connect THIS claim's actual values for the most important "
    "features to the model's output, clearly hedged (global importance, not proven cause).\n"
    "(4) Suggested checks for manual review — concrete, claim-specific things the adjuster could "
    "verify given the flagged features/values (documentation, data quality, corroboration).\n"
    "(5) Reminder — one sentence that the final decision rests with the adjuster.\n"
)


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
