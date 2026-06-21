"""LLM service configuration: model, generation params, and prompts.

The single source of truth is ``llm_config.yaml`` in this package. It is
version-controlled together with the prompts so that, for any release (git tag),
the exact model and prompt text are recoverable. This module loads that YAML and
exposes:

* ``VERSION`` — the config version (bump on model/prompt changes)
* ``MODEL``   — the OpenAI model name
* ``get_prompt(key)``      -> (system, human) for a prompt key
* ``get_generation(key)``  -> {"temperature", "max_tokens"} for a prompt key
* ``TIMEOUT``              — request timeout (seconds)

Prompt keys: ``model_explanation``, ``prediction_adjuster``, ``prediction_customer``.

Back-compat constants (``SYSTEM_PROMPT``/``HUMAN_TEMPLATE`` and the ADJUSTER_*/
CUSTOMER_* variants) are re-exported so existing callers keep working.
"""
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent / "llm_config.yaml"
_cfg = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))

VERSION: str = str(_cfg["version"])
MODEL: str = _cfg["model"]
TIMEOUT: int = int(_cfg.get("defaults", {}).get("timeout", 60))


def get_prompt(key: str) -> tuple[str, str]:
    """Return (system, human) prompt templates for a prompt key."""
    p = _cfg["prompts"][key]
    return p["system"], p["human"]


def get_generation(key: str) -> dict:
    """Return generation params ({temperature, max_tokens}) for a prompt key."""
    return dict(_cfg["generation"][key])


# --- Back-compat constants (sourced from the YAML) -----------------------
SYSTEM_PROMPT, HUMAN_TEMPLATE = get_prompt("model_explanation")
ADJUSTER_SYSTEM_PROMPT, ADJUSTER_HUMAN_TEMPLATE = get_prompt("prediction_adjuster")
CUSTOMER_SYSTEM_PROMPT, CUSTOMER_HUMAN_TEMPLATE = get_prompt("prediction_customer")

__all__ = [
    "VERSION", "MODEL", "TIMEOUT", "get_prompt", "get_generation",
    "SYSTEM_PROMPT", "HUMAN_TEMPLATE",
    "ADJUSTER_SYSTEM_PROMPT", "ADJUSTER_HUMAN_TEMPLATE",
    "CUSTOMER_SYSTEM_PROMPT", "CUSTOMER_HUMAN_TEMPLATE",
]
