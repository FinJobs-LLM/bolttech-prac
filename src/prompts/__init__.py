"""Prompt templates for the LLM explanation features, grouped by audience.

Split out of ``llm_explain.py`` so prompts are easy to version-control, review
and maintain independently of the generation logic. Each submodule owns the
system + human templates for one feature:

* :mod:`prompts.model_explanation` — model-level explanation (``GET /explain``)
* :mod:`prompts.adjuster`          — per-prediction explanation for a claims adjuster
* :mod:`prompts.customer`          — per-prediction explanation for a customer

The constants are re-exported here for convenience; ``llm_explain`` imports them
from the individual submodules.
"""
from prompts.adjuster import ADJUSTER_HUMAN_TEMPLATE, ADJUSTER_SYSTEM_PROMPT
from prompts.customer import CUSTOMER_HUMAN_TEMPLATE, CUSTOMER_SYSTEM_PROMPT
from prompts.model_explanation import HUMAN_TEMPLATE, SYSTEM_PROMPT

__all__ = [
    "SYSTEM_PROMPT",
    "HUMAN_TEMPLATE",
    "ADJUSTER_SYSTEM_PROMPT",
    "ADJUSTER_HUMAN_TEMPLATE",
    "CUSTOMER_SYSTEM_PROMPT",
    "CUSTOMER_HUMAN_TEMPLATE",
]
