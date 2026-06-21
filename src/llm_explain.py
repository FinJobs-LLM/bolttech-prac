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
    "Rules: only use the numbers provided — never invent metrics or features; keep it to "
    "roughly 150-220 words; use short paragraphs or bullets; do not give financial advice; "
    "end with one sentence on a key limitation."
)

HUMAN_TEMPLATE = (
    "Explain this served model and what drives its predictions.\n\n"
    "MODEL\n"
    "- Name: {model_name} ({stage})\n"
    "- Version: {model_version}\n"
    "- Imbalance strategy: {imbalance_strategy}\n"
    "- Decision threshold: {threshold}\n\n"
    "TEST METRICS (held-out)\n{metrics_block}\n\n"
    "TOP FEATURE IMPORTANCES (higher = more influential)\n{features_block}\n\n"
    "Write: (1) what the model does and how to read its decisions at the threshold, "
    "(2) how good it is, emphasising PR-AUC / recall / precision for Declined over accuracy "
    "and why, (3) which features drive it and a plausible, clearly-hedged intuition for the "
    "top few, and (4) one key limitation."
)


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


def _fmt_features(items: list, top_n: int = 12) -> str:
    rows = sorted(items, key=lambda x: x.get("importance", 0), reverse=True)[:top_n]
    return "\n".join(f"- {r['feature']}: {float(r['importance']):.2f}" for r in rows) or "- (none)"


def generate_model_explanation(model_info: dict, feature_importance: list,
                               temperature: float = 0.2) -> str:
    """Call gpt-4o-mini via LangChain and return the explanation text.

    Raises RuntimeError if OPENAI_API_KEY is missing.
    """
    if not explanation_available():
        raise RuntimeError("OPENAI_API_KEY is not set on the server.")

    # Imported lazily so the rest of the API works without langchain installed.
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=MODEL_NAME, temperature=temperature, max_tokens=450, timeout=60)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_TEMPLATE)]
    )
    chain = prompt | llm
    resp = chain.invoke({
        "model_name": model_info.get("model_name") or model_info.get("model") or "unknown",
        "stage": model_info.get("stage", ""),
        "model_version": model_info.get("model_version") or "n/a",
        "imbalance_strategy": model_info.get("imbalance_strategy", "n/a"),
        "threshold": model_info.get("threshold", "n/a"),
        "metrics_block": _fmt_metrics(model_info.get("test_metrics", {}) or {}),
        "features_block": _fmt_features(feature_importance or []),
    })
    return resp.content.strip()
